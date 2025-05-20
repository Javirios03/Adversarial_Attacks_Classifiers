from models.AllConv import AllConv
from models.NiN import NiN
from models.VGG16 import VGG16

from torchsummary import summary
from ptflops import get_model_complexity_info
import time
import torch