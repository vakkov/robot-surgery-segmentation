"""
Squeeze and Excitation Module
*****************************

Collection of squeeze and excitation classes where each can be inserted as a block into a neural network architechture

    1. `Channel Squeeze and Excitation <https://arxiv.org/abs/1709.01507>`_
    2. `Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_
    3. `Channel and Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_

"""

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import ConvABN_GAU

class GAUModule(nn.Module):
    def __init__(self,in_channels, out_channels, norm_act=InPlaceABNSync):
        super(GAUModule, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            #TODO: REPLACE WITH INPLACE BATCH NORM

            Conv2dBn(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            #ConvABN_GAU(out_channels, out_channels, kernel_size=1, stride=1, padding=0, activation ="elu")
            nn.Sigmoid()
        )
        
        #self.conv2 = Conv2dBnRelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvABN_GAU(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    # x: low level feature
    # y: high level feature
    def forward(self,x,y):
        h,w = x.size(2),x.size(3)
        #y_up = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y)
        y_up = nn.Upsample(size=(h, w), mode='nearest', align_corners=True)(y)
        x = self.conv2(x)
        y = self.conv1(y)
        z = torch.mul(x, y)
        
        return y_up + z


class GAUModulev2(nn.Module):
    def __init__(self,in_channels, out_channels, norm_act=InPlaceABNSync):
        super(GAUModule, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            #ConvABN_GAU(out_channels, out_channels, kernel_size=1, stride=1, padding=0, activation ="elu")
            nn.Sigmoid()
        )
        
        self.conv2 = ConvABN_GAU(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    # x: low level feature
    # y: high level feature
    def forward(self,x,y):
        y = self.conv1(x)
        z = torch.mul(x, y)
        z = self.conv2(z)

        return z


class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, pooling = False):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensorx
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Add pooling layer
        if pooling :
            #input_tensor = self.avg_pool(input_tensor).view(batch_size, num_channels)
            input_tensor = self.avg_pool(input_tensor)
        # Average along each channel
        # print('input_tensor')
        # print(input_tensor.size())
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # print('squeeze_tensor')
        # print(squeeze_tensor.size())        

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        # print('output_tensor')
        # print(output_tensor.size())
        return output_tensor


class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """

        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """

        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, a, b))
        # print('output_tensor')
        # print(output_tensor.size())

        return output_tensor


class ChannelSpatialSELayer(nn.Module):
    """
    Re-implementation of concurrent spatial and channel squeeze & excitation:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor, pooling):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """

        #TODO: SHOULDN'T THIS ADD THE TWO TENSORS?
        output_tensor = torch.max(self.cSE(input_tensor, pooling), self.sSE(input_tensor))
        return output_tensor


class SELayer(nn.Module):
    """
    Enum restricting the type of SE Blockes available. So that type checking can be adding when adding these blockes to
    a neural network::

        if self.se_block_type == se.SELayer.CSE.value:
            self.SELayer = se.ChannelSpatialSELayer(params['num_filters'])

        elif self.se_block_type == se.SELayer.SSE.value:
            self.SELayer = se.SpatialSELayer(params['num_filters'])

        elif self.se_block_type == se.SELayer.CSSE.value:
            self.SELayer = se.ChannelSpatialSELayer(params['num_filters'])
    """
    # NONE = 'NONE'
    # CSE = 'CSE'
    # SSE = 'SSE'
    # CSSE = 'CSSE'
   
    def __init__(self, Enum, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(SELayer, self).__init__()
        if 'CSE' == Enum :
            self.SELayer = ChannelSELayer(num_channels, reduction_ratio=2)
        if 'SSE' == Enum :
            self.SELayer = SpatialSELayer(num_channels)
        if 'CSSE' == Enum :
            self.SELayer = ChannelSpatialSELayer(num_channels, reduction_ratio=2)

    def forward(self, input_tensor, pooling):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        #output_tensor = self.SELayer.forword(input_tensor)
        output_tensor = self.SELayer(input_tensor, pooling)
        return output_tensor
    
