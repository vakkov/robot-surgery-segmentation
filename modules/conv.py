from torch import nn
import torch
from torchvision import models
import torchvision
from torch.nn import functional as F
from modules.inplace_abn.abn import InPlaceABNSync


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


def DoubleConv(filters, kernel_size, initializer='glorot_uniform'):

    def layer(x):

        x = Conv2D(filters, kernel_size, padding='same', use_bias=False, kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, kernel_size, padding='same', use_bias=False, kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x

    return layer



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

    #def __init__(self, in_ch: int, out_ch: int, kernel_size=3, stride=1, padding=0, bias=False, norm_act=InPlaceABNSync, activation="leaky_relu"):
    def __init__(self, in_ch: int, out_ch: int, kernel_size=3, stride=1, padding=0, bias=False, norm_act=InPlaceABNSync):    
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.norm_act = norm_act(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm_act(x)
        return x

class ConvTransABN_GAU(nn.Module):

    #def __init__(self, in_ch: int, out_ch: int, kernel_size=3, stride=1, padding=0, bias=False, norm_act=InPlaceABNSync, activation="leaky_relu"):
    def __init__(self, in_ch: int, out_ch: int, kernel_size=3, stride=1, padding=0, bias=False, norm_act=InPlaceABNSync):    
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.norm_act = norm_act(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm_act(x)
        return x



# def DoubleConv(filters, kernel_size, initializer='glorot_uniform'):

#     def layer(x):

#         x = Conv2D(filters, kernel_size, padding='same', use_bias=False, kernel_initializer=initializer)(x)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         x = Conv2D(filters, kernel_size, padding='same', use_bias=False, kernel_initializer=initializer)(x)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)

#         return x

#     return layer        

# def UpSampling2D_block(filters, kernel_size=(3, 3), upsample_rate=(2, 2), interpolation='bilinear',
#                        initializer='glorot_uniform', skip=None):
#     def layer(input_tensor):

#         x = UpSampling2D(size=upsample_rate, interpolation=interpolation)(input_tensor)

#         if skip is not None:
#             x = Concatenate()([x, skip])

#         x = DoubleConv(filters, kernel_size, initializer=initializer)(x)

#         return x
#     return layer        

# def Conv2DTranspose_block(filters, kernel_size=(3, 3), transpose_kernel_size=(2, 2), upsample_rate=(2, 2),
#                           initializer='glorot_uniform', skip=None):
#     def layer(input_tensor):

#         x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate, padding='same')(input_tensor)

#         if skip is not None:
#             x = Concatenate()([x, skip])

#         x = DoubleConv(filters, kernel_size, initializer=initializer)(x)

#         return x

#     return layer        

# def ConvBnRelu(num_in, num_out, kernel_size, stride=1, padding=0, bias=False):
#     return nn.Sequential(
#         nn.Conv2d(num_in, num_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
#         nn.BatchNorm2d(num_out, num_out),
#         nn.ReLU(inplace=True),
# )        