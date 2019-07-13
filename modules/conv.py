from torch import nn
import torch
from torchvision import models
import torchvision
from torch.nn import functional as F
from modules.bn import InPlaceABNSync


# def conv3x3(in_, out):
#     return nn.Conv2d(in_, out, 3, padding=1)
def conv3x3(in_, out, bias=True):
    return nn.Conv2d(in_, out, 3, padding=1, bias=bias)

# class ConvRelu(nn.Module):
#     def __init__(self, in_: int, out: int):
#         super(ConvRelu, self).__init__()
#         self.conv = conv3x3(in_, out)
#         self.activation = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.activation(x)
#         return x



class ConvRelu(nn.Module):
    
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class ConvABN(nn.Module):

    def __init__(self, in_: int, out: int, norm_act=InPlaceABNSync):
        super().__init__()
        self.conv = conv3x3(in_, out, bias=False)
        self.norm_act = norm_act(out)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm_act(x)
        return x

class ConvABN_GAU(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, kernel_size=3, stride=1, padding=0, bias=False, norm_act=InPlaceABNSync):
        super().__init__()
        #self.conv = conv3x3(in_, out, bias=False)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.norm_act = norm_act(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm_act(x)
        return x

# def ConvBnRelu(num_in, num_out, kernel_size, stride=1, padding=0, bias=False):
#     return nn.Sequential(
#         nn.Conv2d(num_in, num_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
#         nn.BatchNorm2d(num_out, num_out),
#         nn.ReLU(inplace=True),
# )        