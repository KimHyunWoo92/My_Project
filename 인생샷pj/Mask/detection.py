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

### detection ####


