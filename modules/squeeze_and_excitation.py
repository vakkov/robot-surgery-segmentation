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


class FPAModule2(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super(FPAModule2, self).__init__()
        
        # global pooling branch
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvABN_GAU(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )
        
        # midddle branch
        self.mid = nn.Sequential(
            ConvABN_GAU(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )
        
        self.down1 = ConvABN_GAU(in_ch, 1, kernel_size=7, stride=2, padding=3)
        
        self.down2 = ConvABN_GAU(1, 1, kernel_size=5, stride=2, padding=2)
        
        self.down3 = nn.Sequential(
            ConvABN_GAU(1, 1, kernel_size=3, stride=2, padding=1),
            ConvABN_GAU(1, 1, kernel_size=3, stride=1, padding=1),
        )
        
        self.conv2 = ConvABN_GAU(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv1 = ConvABN_GAU(1, 1, kernel_size=7, stride=1, padding=3)
    
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        b1 = self.branch1(x)
        b1 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(b1)
        
        mid = self.mid(x)
        
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = nn.Upsample(size=(h // 4, w // 4), mode='bilinear', align_corners=True)(x3)
        
        x2 = self.conv2(x2)
        x = x2 + x3
        x = nn.Upsample(size=(h // 2, w // 2), mode='bilinear', align_corners=True)(x)
        
        x1 = self.conv1(x1)
        x = x + x1
        x = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(x)
        
        x = torch.mul(x, mid)
        x = x + b1
        return x

class GAUModule(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(GAUModule, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            #TODO: REPLACE WITH INPLACE BATCH NORM

            Conv2dBn(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            #ConvABN_GAU(out_channels, out_channels, kernel_size=1, stride=1, padding=0, activation ="relu")
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
    def __init__(self,in_channels, out_channels):
        super(GAUModulev2, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
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

class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        # Global Attention Upsample
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low, fm_mask=None):
        """
        Use the high level features with abundant catagory information to weight the low level features with pixel
        localization information. In the meantime, we further use mask feature maps with catagory-specific information
        to localize the mask position.
        :param fms_high: Features of high level. Tensor.
        :param fms_low: Features of low level.  Tensor.
        :param fm_mask:
        :return: fms_att_upsample
        """
        b, c, h, w = fms_high.shape

        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)# arlog, when the spatial size HxW = 1x1, the BN cannot be used.
        fms_high_gp = self.relu(fms_high_gp)

        # fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp
        if self.upsample:
            out = self.relu(
                self.bn_upsample(self.conv_upsample(fms_high)) + fms_att)
        else:
            out = self.relu(
                self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)
        return out



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
        self.sSE = SpatialSELayer(num_channels)
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)

    def forward(self, input_tensor, pooling):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """

        #TODO: SHOULDN'T THIS ADD THE TWO TENSORS?
        # print("input: ", input_tensor.size())
        # print("cse: ", self.cSE(input_tensor, pooling).size())
        # print("sse: ", self.sSE(input_tensor).size())
        output_tensor = torch.max(self.cSE(input_tensor, pooling), self.sSE(input_tensor))
        #output_tensor = self.sSE(input_tensor) + self.cSE(input_tensor, pooling)
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
        #output_tensor = self.SELayer.forward(input_tensor)
        output_tensor = self.SELayer(input_tensor, pooling)
        return output_tensor
    





class FPA(nn.Module):
    def __init__(self, channels=2048):
        """
        Feature Pyramid Attention
        :type channels: int
        """
        super(FPA, self).__init__()
        channels_mid = int(channels/4)

        self.channels_cond = channels

        # Master branch
        self.conv_master = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_master = nn.BatchNorm2d(channels)

        # Global pooling branch
        self.conv_gpb = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_gpb = nn.BatchNorm2d(channels)

        # C333 because of the shape of last feature maps is (16, 16).
        self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(7, 7), stride=2, padding=3, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=2, padding=2, bias=False)
        self.bn2_1 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(channels_mid)

        self.conv7x7_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(7, 7), stride=1, padding=3, bias=False)
        self.bn1_2 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=1, padding=2, bias=False)
        self.bn2_2 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(channels_mid)

        # Convolution Upsample
        self.conv_upsample_3 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_3 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_2 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_2 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_1 = nn.ConvTranspose2d(channels_mid, channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_1 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        # Master branch
        x_master = self.conv_master(x)
        x_master = self.bn_master(x_master)

        # Global pooling branch
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)

        # Branch 1
        x1_1 = self.conv7x7_1(x)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv7x7_2(x1_1)
        x1_2 = self.bn1_2(x1_2)

        # Branch 2
        x2_1 = self.conv5x5_1(x1_1)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)
        x2_2 = self.bn2_2(x2_2)

        # Branch 3
        x3_1 = self.conv3x3_1(x2_1)
        x3_1 = self.bn3_1(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv3x3_2(x3_1)
        x3_2 = self.bn3_2(x3_2)

        # Merge branch 1 and 2
        x3_upsample = self.relu(self.bn_upsample_3(self.conv_upsample_3(x3_2)))
        x2_merge = self.relu(x2_2 + x3_upsample)
        x2_upsample = self.relu(self.bn_upsample_2(self.conv_upsample_2(x2_merge)))
        x1_merge = self.relu(x1_2 + x2_upsample)

        x_master = x_master * self.relu(self.bn_upsample_1(self.conv_upsample_1(x1_merge)))

        #
        out = self.relu(x_master + x_gpb)

        return out