import datetime
import math
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import torchvision

"""
아래 파일들은 가지고있는 폴더에서 불러오는 것.
import utils
import visualize
from nms.nms_wrapper import nms
from roialign.roi_align.crop_and_resize import CropAndResizeFunction
"""


### loss ####

# rpn loss 계산을 위해, rpn에서 나온 박스의 위치와, 클래스 판별 loss 필요
def rpn_class_loss(rpn_boxes,rpn_class_logits):
    """
    RPN anchor classifier loss.
    rpn_boxes: [batch, anchors, 1]. Anchor match type. 1 = positive
               -1=negative, 0=neutral anchor.
               
    rpn_class: [batch, anchors, 2]. RPN classifier logits for FG/BG
    FG: 객체가 존재 할 확률이 높은 anchor
    BG: 객체가 존재하지 않을 확률이 높은 anchor
    
    """
    #마지막 차원 단순화,batch 값이 없으므로 squeeze는 1로
    rpn_boxes = rpn_boxes.squeeze(1)
    
    #앵커클래스를 받고, -1/1의 일치값을 0/1로 변환
    anchor_class = (rpn_boxes == 1).int()
    
    #loss에 쓰일 Positive 앵커와 Negative 앵커만 선별, 중립 앵커는 제외
    #in_anchor는 0이 아닌값만 가지고 있기때문에, 손실값을 측정할때 필요
    in_anchor = torch.nonzero(rpn_boxes !=0)
    
    # 0이 아닌 값만 갖는 in_anchor를 가지고, 배치와 앵커를 판별하는 것.
    rpn_class_logits = rpn_class_logits[in_anchor.data[:,0],:]
    anchor_class = anchor_class[in_anchor.data[:,0]]
    
    #크로스 엔트로피 사용
    loss = F.cross_entropy(rpn_class_logits,anchor_class)
    
    return loss


def rpn_bbox_loss(target_bbox,rpn_match,rpn_bbox):
    """
    target_box: [max positive anchors,(dy, dx, log(dh), log(dw))].
                 Uses 0 padding to fill in unsed bbox deltas.
                
    rpn_match: [batch,anchors,1]. Anchor match type. 1=positive,
                -1 = negative, 0 =neutral anchor.
    
    rpn_bbox: [batch,anchors, (dy,dx,log(dh),log(dw))]
    
    단일 피쳐맵일 경우 batch의 값은 없을 것으로 생각됨, 위에서 말하는 batch는 FPN을 거쳐 나오는
    각 Conv 층에서 나오는 피쳐맵에 각각의 anchor가 있기 때문. 따라서 FPN을 거쳐서 나오는 batch
    아래의 값이 있을 것임. 
    ex) batch = (batch,num_boxes)
    따라서, 단일 피쳐맵일 경우, batch는 num_boxes의 값을 가진다고 보면 됨.
                
    """
    
    # 차원 단순화
    rpn_match = rpn_match.squeeze(1)
    
    # 앵커박스가 positive인것과 negative인 값만 추출
    in_anchor = torch.nonzero(rpn_match==1)
    
    #loss 계산을 위해 값을 가진 앵커박스만 추출 후 갱신신
    rpn_bbox = rpn_bbox[in_anchor.data[:,0]]
    
    #target box를 bbox와 동일한 사이즈로 변경
    target_bbox = target_bbox[0,:rpn_bbox.size()[0]]
    
    #Smooth L1 loss 사용, L1 loss를 사용하지 않는 것은,
    #L1 loss는 학습과정에서 기울기가 정의되지 않는 부분에서 불안정 하지만,
    #Smooth L1 loss는 작은 오차에 대해서 미분가능성을 확보해 안정성을 획득가능.
    loss = F.smooth_l1_loss(rpn_bbox, target_bbox)
    
    
def mrcnn_class_loss(target_class_ids, pred_class_logits):
    """
    Mask RCNN의 Head부분에 분류 loss.
    target_class_idx: [num_boxes,num_rois]. Integer class IDs. Uses zero
                       padding to fill in the array
                       
    pred_class_logits: [num_boxes,num_rois,num_classes]
    
    rois:  이미지 내에서 특정 관심 영역을 나타내며, 이는 주로 bbox로 정의됨.
    
    """
    
    #Loss
    if target_class_ids.size():
        loss = F.cross_entropy(pred_class_logits,target_class_ids.int())
        
    else:
        loss = Variable(torch.FloatTensor([0]),requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()
            
    return loss

    
def mrcnn_bbox_loss(target_bbox,target_class_ids,pred_bbox):
    """
    Loss for Mask R-CNN bounding box refinement.
    
    target_box: [batch,num_rois,(dy,dx,log(dh),log(dw))]
    target_class_ids: [batch,num_rois]. Interger class IDs.
    pred_bbox: [batch, num_rois,num_classes,(dy,dx,log(dh),log(dw))]
    
    위 값들은 FPN이 있을때의 각 변수의 값이지만, 단일 피쳐맵을 거칠 경우, batch부분은
    이미지 인덱스로 생각해야함.
    
    """
    # positive ROIs 만 loss에 사용. 올바른 클래스 id만 추출
    if target_class_ids.size():
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix].int()
        
        #indices는 지수라는 뜻, 올바른 클래스만 1차원으로 만들어서 가지고 있겠다.
        indices = torch.stack((positive_roi_ix,positive_roi_class_ids),dim=1)
        
        #손실계산을 위해 예측값과 실제값만 추출
        target_bbox = target_bbox[indices[:,0],:]
        pred_bbox = pred_bbox[indices[:,0],indices[:,1],:]
        
        #Smooth L1 loss 사용
        loss = F.smooth_l1_loss(pred_bbox,target_bbox)
        
    else:
        loss = Variable(torch.FloatTensor([0]),requires_grad=False)
        if target_class_ids.is_cuda():
            loss = loss.cuda()
            
            
    return loss


def mrcnn_mask_loss(target_mask,target_class_ids,pred_mask):
    """
    Mask binary cross_entropy loss for the mask head.
    target_mask: [batch,num_rois,height,width].
                    A float32 tensro of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch,num_rois]. Interger class IDs. Zero padded.
    pred_mask: [batch,proposals,height,width,num_classes] float32 tensor with
                values from 0 to 1
                
    
    """
    
    #loss에는 예측값과 실제값의 차이만으로 계산해야 함.
    if target_class_ids.size():
        positive_ix = torch.nonzero(target_class_ids>0)[:,0]
        positive_class_ids = target_class_ids[positive_ix].int()
        indices = torch.stack((positive_ix,positive_class_ids),dim=1)
        
        y_true = target_mask[indices[:,0],:,:]
        y_pred = pred_mask[indices[:,0],:,:,indices[:,1]]
        
        #이진분류
        loss = F.binary_cross_entropy(y_pred,y_true)
    
    else:
        loss = Variable(torch.FloatTensor([0]),requires_grad=False)
        if target_class_ids.is_cuda():
            loss = loss.cuda()
            
    return loss

def all_losses(rpn_match,rpn_bbox,rpn_class_logits,rpn_pred_bbox,target_class_ids,
               mrcnn_class_logits,target_deltas,mrcnn_bbox,target_mask,mrcnn_mask):
    
    comput_rpn_class_loss = rpn_class_loss(rpn_match,rpn_class_logits)
    comput_rpn_bbox_loss = rpn_bbox_loss(rpn_bbox,rpn_match,rpn_pred_bbox)
    comput_mrcnn_class_loss = mrcnn_class_loss(target_class_ids,mrcnn_class_logits)
    comput_mrcnn_bbox_loss = mrcnn_bbox_loss(target_deltas,target_class_ids,mrcnn_bbox)
    comput_mrcnn_mask_loss = mrcnn_mask_loss(target_mask,target_class_ids,mrcnn_mask)
    
    return [comput_rpn_class_loss,comput_rpn_bbox_loss,comput_mrcnn_class_loss,
            comput_mrcnn_bbox_loss,comput_mrcnn_mask_loss]
    
### loss end ###


 
    
    
    