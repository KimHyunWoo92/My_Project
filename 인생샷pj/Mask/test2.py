import torch
import torch.nn.functional as F
from functorch.dim import dims
import math
import torch.nn as nn
from torchvision import models
from torchvision.ops import roi_align
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np



def bilinear_interpolate(input, height, width, y, x, ymask, xmask):
    
    y = y.clamp(min=0)
    x = x.clamp(min=0)
    y_low = y.int()
    x_low = x.int()
    y_high = torch.where(y_low >= height - 1, height - 1, y_low + 1)
    y_low = torch.where(y_low >= height - 1, height - 1, y_low)
    y = torch.where(y_low >= height - 1, y.to(input.dtype), y)

    x_high = torch.where(x_low >= width - 1, width - 1, x_low + 1)
    x_low = torch.where(x_low >= width - 1, width - 1, x_low)
    x = torch.where(x_low >= width - 1, x.to(input.dtype), x)

    ly = y - y_low
    lx = x - x_low
    hy = 1. - ly
    hx = 1. - lx
    
    def masked_index(y, x):
        y = torch.where(ymask, y, 0)
        x = torch.where(xmask, x, 0)
        return input[y, x]

    v1 = masked_index(y_low, x_low)
    v2 = masked_index(y_low, x_high)
    v3 = masked_index(y_high, x_low)
    v4 = masked_index(y_high, x_high)
    w1 = hy * hx
    w2 = hy * lx
    w3 = ly * hx
    w4 = ly * lx

    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    return val

def roi_align(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned):
    _, _, height, width = input.size()

    n, c, ph, pw = dims(4)
    ph.size = pooled_height
    pw.size = pooled_width
    offset_rois = rois[n]
    roi_batch_ind = offset_rois[0].int()
    offset = 0.5 if aligned else 0.0
    roi_start_w = offset_rois[1] * spatial_scale - offset
    roi_start_h = offset_rois[2] * spatial_scale - offset
    roi_end_w = offset_rois[3] * spatial_scale - offset
    roi_end_h = offset_rois[4] * spatial_scale - offset

    roi_width = roi_end_w - roi_start_w
    roi_height = roi_end_h - roi_start_h
    if not aligned:
        roi_width = torch.clamp(roi_width, min=1.0)
        roi_height = torch.clamp(roi_height, min=1.0)

    bin_size_h = roi_height / pooled_height
    bin_size_w = roi_width / pooled_width

    offset_input = input[roi_batch_ind][c]

    roi_bin_grid_h = sampling_ratio if sampling_ratio > 0 else torch.ceil(roi_height / pooled_height)
    roi_bin_grid_w = sampling_ratio if sampling_ratio > 0 else torch.ceil(roi_width / pooled_width)

    count = torch.clamp(roi_bin_grid_h * roi_bin_grid_w, min=1)

    iy, ix = dims(2)
    
    iy.size = height  # < roi_bin_grid_h
    ix.size = width  # < roi_bin_grid_w
    
    
    y = roi_start_h + ph * bin_size_h + (iy + 0.5) * bin_size_h / roi_bin_grid_h
    x = roi_start_w + pw * bin_size_w + (ix + 0.5) * bin_size_w / roi_bin_grid_w
    ymask = iy < roi_bin_grid_h
    xmask = ix < roi_bin_grid_w
  
    val = bilinear_interpolate(offset_input, height, width, y, x, ymask, xmask)
    val = torch.where(ymask, val, 0)
    val = torch.where(xmask, val, 0)
    output = val.sum((iy, ix))
    output /= count

    return output.order(n, c, ph, pw)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        self.avg_pooling = nn.AdaptiveAvgPool2d((28,28))

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.avg_pooling(x)

        return x

simple_cnn = CNN()

# coco 이미지 불러오기
image_path = "COCO_train2014_000000000030.jpg"
image = Image.open(image_path)

#텐서 변환
preprocess = transforms.Compose([transforms.ToTensor(),])

input_image = preprocess(image)
input_image = input_image.unsqueeze(0)

output_image = simple_cnn(input_image)

# 임의의 피쳐맵 생성
features = output_image

rois = torch.tensor([
    [0,1,1,5,5],
    [0,15,15,20,20],
], dtype=torch.float)


output_size = (7, 7)
spatial_scale = 1.0 / 4.0

# Call the roi_align function
pooled_features = roi_align(features, rois, spatial_scale, output_size[0], output_size[1], -1, False)

print(pooled_features.sum())

from torchvision.ops import roi_align as roi_align_torchvision

print(roi_align_torchvision(features, rois, output_size, spatial_scale).sum())
