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


def roi_align(inputs,pooling_size,image_shape):
    """
    Inputs:
    - boxes: [batch,num_boxes,(y1,x1,y2,x2)] in normalized coordinates.
    - featuremap: feature map from RPN
    
    - pooling_size: [height,weight] of the output pooled regions. Usually [7,7]
    - image_size: [height,weight,channels]. Shape of input image in pexels
    """

    
    # crop boxes [batch,num_boxes,(y1, x1, y2, x2)] in normalized corrds
    boxes = inputs[0]

    #Feature map. RPN을 통과한 피쳐맵을 받는 것.
    feature_map = inputs[1:]

    #ROI area
    y1,x1,y2,x2 = boxes[1:]
    h = y2 - y1
    w = x2 - x1

    #좌표 정규화, 예를들면 224x224 ROI(in piexels) map to P4(예시에 있는 FPN layer에 P4)
    images_area = Variable(torch.FloatTensor([float(image_shape[0]*image_shape[1])]), requires_grad=False)
    if boxes.is_cuda:
        images_area = images_area.cuda()
    #Conv2D 이후 바로 나온 feature map에 영역제안을 받기때문에, 레벨박스 필요x
    pooled = []
    
    ix = torch.nonzero()[:,0]
    all_boxes = boxes[ix.data,:]
    all_boxes = all_boxes.detach()
    
    ind = torch.tensor(torch.zero(all_boxes.size()[0]),requires_grad=False).int()
    if all_boxes.is_coda():
        ind = ind.cuda()
    
    feature_map = feature_map.unsqueeze(0)
    # pooled_feature = CropAndResizeFunction(pool_size, pool_size, 0)(feature_map,all_boxes, ind)