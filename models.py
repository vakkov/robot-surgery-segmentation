from torch import nn
import torch
from torchvision import models
import torchvision
from torch.nn import functional as F
from torch.nn import Sequential
from collections import OrderedDict
#from modules.bn import InPlaceABN
from modules.inplace_abn.abn import InPlaceABNSync
from modules.misc import GlobalAvgPool2d
from modules.conv import ConvRelu, ConvABN, ConvABN_GAU
from modules.squeeze_and_excitation import SELayer, GAUModulev2, GAU, FPA #, AttentionSpatialSELayer

from modules.wider_resnet import WiderResNet
from pathlib import Path
from torchsummary import summary

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class DecoderBlock_U(nn.Module):
    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock_U, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels)          
            )

    def forward(self, x):
        return self.block(x)

class Upsample_nearest(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2, mode='nearest')

class DecoderBlock(nn.Module):
    
    def __init__(self, in_channels, middle_channels, out_channels, norm_act=InPlaceABNSync):
        
        super().__init__()
        
        self.block = nn.Sequential(
            Upsample_nearest(),
            ConvABN(in_channels, middle_channels, norm_act=norm_act),
            ConvABN(middle_channels, out_channels, norm_act=norm_act)
        )

    def forward(self, x):
        return self.block(x)


#class DecoderRasV3(nn.Module):
    
    #def __init__(self, in_channels, middle_channels, out_channels, is_deconv = False, norm_act=InPlaceABNSync):
        
        #super().__init__()

        # self.first_1x1 = ConvABN_GAU(in_channels, middle_channels, 1, norm_act=norm_act)
        # self.conv_upsample = ConvTransABN_GAU(middle_channels, middle_channels, kernel_size=4, stride=2, padding=1, bias=False)
        # self.second_1x1 = ConvABN_GAU(middle_channels, out_channels, 1, norm_act=norm_act)
    # def forward(self, x):

    #     x = self.first_1x1(x)
    #     x = self.conv_upsample(x)
    #     x = self.second_1x1(x)
    #     return x


class DecoderBlockv2(nn.Module):
    
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv = False, norm_act=InPlaceABNSync):
        
        super().__init__()
        
        self.block = nn.Sequential(
            #Upsample_nearest(),
            ConvABN(in_channels, middle_channels, norm_act=norm_act),
            ConvABN(middle_channels, out_channels, norm_act=norm_act)
        )

    def forward(self, x, e=None):
        #print("x before upsampling: ")
        #print(x.shape)
        if e is not None:
            x = torch.cat([x, e], 1)        
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        #print("x after upsampling: ")
        #print(x.shape)
        x = self.block(x)
        #print("x after block: ")
        #print(x.shape)
        return x


class DecoderBlockv3(DecoderBlockv2):
    
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv = False, SELayer_type = 'None', norm_act=InPlaceABNSync):
        
        super(DecoderBlockv3, self).__init__(in_channels, middle_channels, out_channels, SELayer_type , norm_act)
        
        self.se = 'None'
        if SELayer_type != 'None' :
            self.se = SELayer(SELayer_type, out_channels) 

    def forward(self, x, e=None):
        # return self.block(x)
        out = super().forward(x, e)
        if self.se != 'None' :
            out = self.se(out, pooling = True)
            #out = functional.leaky_relu(out, negative_slope=0.01, inplace=True)      
        return out


class DecoderRasV3(nn.Module):
    
    #def __init__(self, in_channels, middle_channels, out_channels, is_deconv = False, norm_act=InPlaceABNSync):
        
        #super().__init__()

        # self.first_1x1 = ConvABN_GAU(in_channels, middle_channels, 1, norm_act=norm_act)
        # self.conv_upsample = ConvTransABN_GAU(middle_channels, middle_channels, kernel_size=4, stride=2, padding=1, bias=False)
        # self.second_1x1 = ConvABN_GAU(middle_channels, out_channels, 1, norm_act=norm_act)
    # def forward(self, x):

    #     x = self.first_1x1(x)
    #     x = self.conv_upsample(x)
    #     x = self.second_1x1(x)
    #     return x

    def __init__(self, in_channels, n_filters, ASELayer = False):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C/4, 2 * H, 2 * W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                          stride=2, padding=1, output_padding=0)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)

        self.ase = None
        if ASELayer==True:
            self.ase = AttentionSpatialSELayer(in_channels, in_channels, upsample = False)

    def forward(self, x, encoder=None):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        if self.ase != 'None' and encoder is not None:
            print("encoder ", encoder.shape)
            out = self.ase(x, encoder)
        return x


class RasNetV3(nn.Module):
    """Variation of the UNet architecture with InplaceABN encoder."""

    def __init__(self, num_classes=1, pretrained=False, num_filters=32, is_deconv=True, num_input_channels=3, Lovasz_softmax = False, **kwargs):
        """
        Args:
            num_classes: Number of output classes.
            num_filters:
            is_deconv:
                True: Deconvolution layer is used in the Decoder block.
                False: Upsampling layer is used in the Decoder block.
            num_input_channels: Number of channels in the input images.
        """
        # super(TernausNetV2, self).__init__()
        super().__init__()

        if 'norm_act' not in kwargs:
            norm_act = InPlaceABNSync
        else:
            norm_act = kwargs['norm_act']

        self.pool = nn.MaxPool2d(2, 2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False,
            count_include_pad=False)
        self.num_classes = num_classes
        filters = [64, 128, 256, 512, 1024]

        encoder = WiderResNet(structure=[3, 3, 6, 3, 1, 1], classes=1000, norm_act=norm_act)
       
        if pretrained:
            model_path = Path(__file__).resolve().parent / 'data' / 'models'
            checkpoint = torch.load((model_path.as_posix() + "/" + 'wide_resnet38_ipabn_lr_256.pth.tar'))

            # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/2
            state_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            encoder.load_state_dict(new_state_dict)
            encoder.bn_out = norm_act(num_input_channels)
            if num_classes != 0:
                encoder.classifier = nn.Sequential(OrderedDict([
                    ("avg_pool", GlobalAvgPool2d()),
                    ("fc", nn.Linear(num_input_channels, num_classes))
                ]))        

        if num_input_channels == 1:
            self.conv1 = encoder.mod1
        else:
            self.conv1 = Sequential(
                OrderedDict([('conv1', nn.Conv2d(num_input_channels, 64, 3, padding=1, bias=False))]))
                #OrderedDict([('conv1', ConvABN_GAU(num_input_channels, 64, kernel_size=3, padding=1, bias=False))]))
        self.conv2 = encoder.mod2
        self.conv3 = encoder.mod3
        self.conv4 = encoder.mod4
        self.conv5 = encoder.mod5

        #self.fpa = FeaturePyramidAttention(1024, 1024)
        self.fpa = FPA(1024)

        #center is the bottleneck block; according to 1803.02579 we should not have squeeze and excitation there.
        #self.center = DecoderBlockv2(1024, num_filters * 8, num_filters * 8, norm_act=norm_act)

        # self.dec5 = DecoderBlockv3(1024 + num_filters * 8, num_filters * 8, num_filters * 8, is_deconv=is_deconv, norm_act=norm_act, SELayer_type='CSSE')
        # self.dec4 = DecoderBlockv3(512 + num_filters * 8, num_filters * 8, num_filters * 8, is_deconv=is_deconv, norm_act=norm_act, SELayer_type='CSSE')
        # self.dec3 = DecoderBlockv3(256 + num_filters * 8, num_filters * 2, num_filters * 2, is_deconv=is_deconv, norm_act=norm_act, SELayer_type='CSSE')
        # self.dec2 = DecoderBlockv3(128 + num_filters * 2, num_filters * 2, num_filters, is_deconv=is_deconv, norm_act=norm_act, SELayer_type='CSSE')
        # self.dec1 = ConvABN(64 + num_filters, num_filters, norm_act=norm_act)

        # self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

        self.center = DecoderRasV3(filters[4], filters[3], ASELayer = False)
        self.dec5 = DecoderRasV3(filters[3], filters[2], ASELayer = True)
        self.dec4 = DecoderRasV3(filters[2], filters[1], ASELayer = True)
        self.dec3 = DecoderRasV3(filters[1], filters[0], ASELayer = True)
        self.dec2 = DecoderRasV3(filters[0], filters[0], ASELayer = True)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        fpa = self.fpa(conv5)
        #center = self.center(nn.functional.max_pool2d(fpa, kernel_size=2, stride=2))
        center = self.center(self.avg_pool(fpa))

        # dec5 = self.dec5(torch.cat([center, conv5], 1))
        # dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        # dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        # dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        # dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        # Decoder with Skip Connections
        print("conv1 ", conv1.shape)
        print("conv2 ", conv2.shape)
        print("conv3 ", conv3.shape)
        print("conv4 ", conv4.shape)
        print("conv5 ", conv5.shape)
        print("FPA ", fpa.shape)
        print("center ", center.shape)


        dec5 = self.dec5(center, conv5)
        dec4 = self.dec4(dec5, conv4)
        dec3 = self.dec3(dec4, conv3)
        dec2 = self.dec2(dec3, conv2)
        dec1 = self.dec1(dec2, conv1)

        # print("conv1 ", conv1.shape)
        # print("conv2 ", conv2.shape)
        # print("conv3 ", conv3.shape)
        # print("conv4 ", conv4.shape)
        # print("conv5 ", conv5.shape)
        # print("FPA ", fpa.shape)
        # print("center ", center.shape)
        print("dec5 ", dec5.shape)
        print("dec4 ", dec4.shape)
        print("dec3 ", dec3.shape)
        print("dec2 ", dec2.shape)
        print("dec1 ", dec1.shape)

        # dec5 = self.dec5(center, conv5)
        # dec4 = self.dec4(dec5, conv4)
        # dec3 = self.dec3(dec4, conv3)
        # dec2 = self.dec2(dec3, conv2)
        # dec1 = self.dec1(torch.cat([dec2, conv1], 1)) 

        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            #x_out = F.log_softmax(f5, dim=1)
            x_out = f5
        else:
            x_out = f5
        return x_out

        # #return self.final(dec1)
        # if self.num_classes > 1:
        #     #x_out = F.log_softmax(self.final(dec1), dim=1)
        #     #x_out = F.softmax(self.final(dec1), dim=1)
        #     x_out = self.final(dec1)
        # else:
        #     x_out = self.final(dec1)

        # return x_out          
    
class RasTerNetV2(nn.Module):
    """Variation of the UNet architecture with InplaceABN encoder."""
    def freeze_encoder(self):
        layers = [self.conv1, self.conv2, self.conv3, self.conv4]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False 

    def __init__(self, num_classes=1, pretrained=False, num_filters=32, is_deconv=True, num_input_channels=3, Lovasz_softmax = False, **kwargs):
        """
        Args:
            num_classes: Number of output classes.
            num_filters:
            is_deconv:
                True: Deconvolution layer is used in the Decoder block.
                False: Upsampling layer is used in the Decoder block.
            num_input_channels: Number of channels in the input images.
        """
        # super(TernausNetV2, self).__init__()
        super().__init__()

        if 'norm_act' not in kwargs:
            norm_act = InPlaceABNSync
        else:
            norm_act = kwargs['norm_act']

        self.pool = nn.MaxPool2d(2, 2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False,
            count_include_pad=False)
        self.num_classes = num_classes

        encoder = WiderResNet(structure=[3, 3, 6, 3, 1, 1], classes=1000, norm_act=norm_act)
       
        if pretrained:
            model_path = Path(__file__).resolve().parent / 'data' / 'models'
            checkpoint = torch.load((model_path.as_posix() + "/" + 'wide_resnet38_ipabn_lr_256.pth.tar'))

            # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/2
            state_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            encoder.load_state_dict(new_state_dict)
            encoder.bn_out = norm_act(num_input_channels)
            if num_classes != 0:
                encoder.classifier = nn.Sequential(OrderedDict([
                    ("avg_pool", GlobalAvgPool2d()),
                    ("fc", nn.Linear(num_input_channels, num_classes))
                ]))        

        if num_input_channels == 1:
            self.conv1 = encoder.mod1
        else:
            self.conv1 = Sequential(
                OrderedDict([('conv1', nn.Conv2d(num_input_channels, 64, 3, padding=1, bias=False))]))
                #OrderedDict([('conv1', ConvABN_GAU(num_input_channels, 64, kernel_size=3, padding=1, bias=False))]))
        self.conv2 = encoder.mod2
        self.conv3 = encoder.mod3
        self.conv4 = encoder.mod4
        self.conv5 = encoder.mod5

        #self.fpa = FeaturePyramidAttention(1024, 1024)
        self.fpa = FPA(channels=1024)

        #center is the bottleneck block; according to 1803.02579 we should not have squeeze and excitation there.
        self.center = DecoderBlockv2(1024, num_filters * 8, num_filters * 8, norm_act=norm_act)

        self.dec5 = DecoderBlockv3(1024 + num_filters * 8, num_filters * 8, num_filters * 8, is_deconv=is_deconv, norm_act=norm_act, SELayer_type='CSSE')
        self.dec4 = DecoderBlockv3(512 + num_filters * 8, num_filters * 8, num_filters * 8, is_deconv=is_deconv, norm_act=norm_act, SELayer_type='CSSE')
        self.dec3 = DecoderBlockv3(256 + num_filters * 8, num_filters * 2, num_filters * 2, is_deconv=is_deconv, norm_act=norm_act, SELayer_type='CSSE')
        self.dec2 = DecoderBlockv3(128 + num_filters * 2, num_filters * 2, num_filters, is_deconv=is_deconv, norm_act=norm_act, SELayer_type='CSSE')
        self.dec1 = ConvABN(64 + num_filters, num_filters, norm_act=norm_act)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        #center = self.center(self.pool(conv5))
        #TODO: POOL THE CONV5 BEFORE FPA??
        fpa = self.fpa(conv5)
        #center = self.center(nn.functional.max_pool2d(fpa, kernel_size=2, stride=2))
        center = self.center(self.avg_pool(fpa))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        # print("dec1 ", dec1.shape)
        # print("dec2 ", dec2.shape)
        # print("dec3 ", dec3.shape)
        # print("dec4 ", dec4.shape)
        # print("dec5 ", dec5.shape)
        # print("center ", center.shape)
        # print("FPA ", fpa.shape)
        # print("conv5 ", conv5.shape)
        # print("conv4", conv4.shape)
        # print("conv3", conv3.shape)
        # print("conv2", conv2.shape)
        # print("conv1", conv1.shape)

        # dec5 = self.dec5(center, conv5)
        # dec4 = self.dec4(dec5, conv4)
        # dec3 = self.dec3(dec4, conv3)
        # dec2 = self.dec2(dec3, conv2)
        # dec1 = self.dec1(torch.cat([dec2, conv1], 1)) 

        #return self.final(dec1)
        if self.num_classes > 1:
            #x_out = F.log_softmax(self.final(dec1), dim=1)
            #x_out = F.softmax(self.final(dec1), dim=1)
            x_out = self.final(dec1)
        else:
            x_out = self.final(dec1)

        return x_out        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RasTerNetV2().to(device)
print(model)

summary(model, (3, 1024, 1280))  
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         #print (name, param.data)
#         print(name)      

class TernausNetV2(nn.Module):
    """Variation of the UNet architecture with InplaceABN encoder."""
    def freeze_encoder(self):
        layers = [self.conv1, self.conv2, self.conv3, self.conv4]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False 

    def __init__(self, num_classes=1, pretrained=False, num_filters=32, is_deconv=False, num_input_channels=3, **kwargs):
    #def __init__(self, num_classes=1, num_filters=32, is_deconv=False, num_input_channels=3, **kwargs):
        """

        Args:
            num_classes: Number of output classes.
            num_filters:
            is_deconv:
                True: Deconvolution layer is used in the Decoder block.
                False: Upsampling layer is used in the Decoder block.
            num_input_channels: Number of channels in the input images.
        """
        # super(TernausNetV2, self).__init__()
        super().__init__()

        if 'norm_act' not in kwargs:
            norm_act = InPlaceABNSync
        else:
            norm_act = kwargs['norm_act']

        self.pool = nn.MaxPool2d(2, 2)
        self.num_classes = num_classes

        encoder = WiderResNet(structure=[3, 3, 6, 3, 1, 1], classes=1000, norm_act=norm_act)
       
        if pretrained:
            model_path = Path(__file__).resolve().parent / 'data' / 'models'
            checkpoint = torch.load((model_path.as_posix() + "/" + 'wide_resnet38_ipabn_lr_256.pth.tar'))

            # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/2
            state_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # new_state_dict = {k: v for k, v in state_dict.items() if k in new_state_dict}

            encoder.load_state_dict(new_state_dict)
            #encoder.fc = nn.Linear(num_input_channels, num_classes)
            encoder.bn_out = norm_act(num_input_channels)
            if num_classes != 0:
                encoder.classifier = nn.Sequential(OrderedDict([
                    ("avg_pool", GlobalAvgPool2d()),
                    ("fc", nn.Linear(num_input_channels, num_classes))
                ]))        


        if num_input_channels == 1:
            self.conv1 = encoder.mod1
        else:
            self.conv1 = Sequential(
                OrderedDict([('conv1', nn.Conv2d(num_input_channels, 64, 3, padding=1, bias=False))]))
        self.conv2 = encoder.mod2
        self.conv3 = encoder.mod3
        self.conv4 = encoder.mod4
        self.conv5 = encoder.mod5
        #Hacky; TODO FIX
        #self.freeze_encoder();

        # self.center = DecoderBlock(1024, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        # self.dec5 = DecoderBlock(1024 + num_filters * 8, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        # self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        # self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 2, num_filters * 2, is_deconv=is_deconv)
        # self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2, num_filters, is_deconv=is_deconv)
        #self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.center = DecoderBlock(1024, num_filters * 8, num_filters * 8, norm_act=norm_act)
        self.dec5 = DecoderBlock(1024 + num_filters * 8, num_filters * 8, num_filters * 8, norm_act=norm_act)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8, num_filters * 8, norm_act=norm_act)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 2, num_filters * 2, norm_act=norm_act)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2, num_filters, norm_act=norm_act)
        self.dec1 = ConvABN(64 + num_filters, num_filters, norm_act=norm_act)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        #return self.final(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)

        return x_out

    def set_fine_tune(self, fine_tune_enabled):
        layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = bool(not fine_tune_enabled)


    #def set_encoder_training_enabled(self, enabled):
    def unfreeze_encoder(self, enabled):
        # First layer is trainable since we use 1-channel image instead of 3-channel
        layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = bool(enabled)   



class DecoderBlockvT(nn.Module):
    
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv = False, SELayer_type = 'None', norm_act=InPlaceABNSync):
        
        super().__init__()
        
        self.block = nn.Sequential(
            Upsample_nearest(),
            ConvABN(in_channels, middle_channels, norm_act=norm_act),
            ConvABN(middle_channels, out_channels, norm_act=norm_act)
        )

        self.se = 'None'
        if SELayer_type != 'None' :
            self.se = SELayer(SELayer_type, out_channels) 

    def forward(self, x):
        # return self.block(x)
        out = self.block(x)
        if self.se != 'None' :
            out = self.se(out, pooling = True)
            #out = functional.leaky_relu(out, negative_slope=0.01, inplace=True)      
        return out
    
class TernausNetOC(nn.Module):
    """Variation of the UNet architecture with InplaceABN encoder."""
    def freeze_encoder(self):
        layers = [self.conv1, self.conv2, self.conv3, self.conv4]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False 

    def __init__(self, num_classes=1, pretrained=False, num_filters=32, is_deconv=False, num_input_channels=3, **kwargs):
    #def __init__(self, num_classes=1, num_filters=32, is_deconv=False, num_input_channels=3, **kwargs):
        """

        Args:
            num_classes: Number of output classes.
            num_filters:
            is_deconv:
                True: Deconvolution layer is used in the Decoder block.
                False: Upsampling layer is used in the Decoder block.
            num_input_channels: Number of channels in the input images.
        """
        # super(TernausNetV2, self).__init__()
        super().__init__()

        if 'norm_act' not in kwargs:
            norm_act = InPlaceABNSync
        else:
            norm_act = kwargs['norm_act']

        self.pool = nn.MaxPool2d(2, 2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False,
            count_include_pad=False)
        self.num_classes = num_classes

        encoder = WiderResNet(structure=[3, 3, 6, 3, 1, 1], classes=1000, norm_act=norm_act)
       
        if pretrained:
            model_path = Path(__file__).resolve().parent / 'data' / 'models'
            checkpoint = torch.load((model_path.as_posix() + "/" + 'wide_resnet38_ipabn_lr_256.pth.tar'))

            # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/2
            state_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # new_state_dict = {k: v for k, v in state_dict.items() if k in new_state_dict}

            encoder.load_state_dict(new_state_dict)
            #encoder.fc = nn.Linear(num_input_channels, num_classes)
            encoder.bn_out = norm_act(num_input_channels)
            if num_classes != 0:
                encoder.classifier = nn.Sequential(OrderedDict([
                    ("avg_pool", GlobalAvgPool2d()),
                    ("fc", nn.Linear(num_input_channels, num_classes))
                ]))        


        if num_input_channels == 1:
            self.conv1 = encoder.mod1
        else:
            self.conv1 = Sequential(
                OrderedDict([('conv1', nn.Conv2d(num_input_channels, 64, 3, padding=1, bias=False))]))
        self.conv2 = encoder.mod2
        self.conv3 = encoder.mod3
        self.conv4 = encoder.mod4
        self.conv5 = encoder.mod5
        #Hacky; TODO FIX
        #self.freeze_encoder();

        # self.center = DecoderBlock(1024, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        # self.dec5 = DecoderBlock(1024 + num_filters * 8, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        # self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        # self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 2, num_filters * 2, is_deconv=is_deconv)
        # self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2, num_filters, is_deconv=is_deconv)
        #self.dec1 = ConvRelu(64 + num_filters, num_filters)
        
        #center = self.center(nn.functional.max_pool2d(fpa, kernel_size=2, stride=2))
        self.fpa = FeaturePyramidAttention(1024, 1024)

        #center is the bottleneck block; according to 1803.02579 we should not have squeeze and excitation there.
        self.center = DecoderBlockvT(1024, num_filters * 8, num_filters * 8, norm_act=norm_act)
        self.dec5 = DecoderBlockvT(1024 + num_filters * 8, num_filters * 8, num_filters * 8, norm_act=norm_act)
        self.dec4 = DecoderBlockvT(512 + num_filters * 8, num_filters * 8, num_filters * 8, norm_act=norm_act)
        self.dec3 = DecoderBlockvT(256 + num_filters * 8, num_filters * 2, num_filters * 2, norm_act=norm_act)
        self.dec2 = DecoderBlockvT(128 + num_filters * 2, num_filters * 2, num_filters, norm_act=norm_act)
        self.dec1 = ConvABN(64 + num_filters, num_filters, norm_act=norm_act)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        fpa = self.fpa(conv5)
        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        #return self.final(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)

        return x_out

    def set_fine_tune(self, fine_tune_enabled):
        layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = bool(not fine_tune_enabled)


    #def set_encoder_training_enabled(self, enabled):
    def unfreeze_encoder(self, enabled):
        # First layer is trainable since we use 1-channel image instead of 3-channel
        layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = bool(enabled)                       

class UNet11(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG11
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.num_classes = num_classes

        self.encoder = models.vgg11(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[3],
                                   self.relu)

        self.conv3 = nn.Sequential(
            self.encoder[6],
            self.relu,
            self.encoder[8],
            self.relu,
        )
        self.conv4 = nn.Sequential(
            self.encoder[11],
            self.relu,
            self.encoder[13],
            self.relu,
        )

        self.conv5 = nn.Sequential(
            self.encoder[16],
            self.relu,
            self.encoder[18],
            self.relu,
        )

        self.center = DecoderBlock_U(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv=True)
        self.dec5 = DecoderBlock_U(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv=True)
        self.dec4 = DecoderBlock_U(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 4, is_deconv=True)
        self.dec3 = DecoderBlock_U(256 + num_filters * 4, num_filters * 4 * 2, num_filters * 2, is_deconv=True)
        self.dec2 = DecoderBlock_U(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv=True)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
            #x_out = F.softmax(self.final(dec1), dim=1)
            #x_out = self.final(dec1)

        else:
            x_out = self.final(dec1)

        return x_out


class UNet16(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG11
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        self.center = DecoderBlock_U(512, num_filters * 8 * 2, num_filters * 8)

        self.dec5 = DecoderBlock_U(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock_U(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlock_U(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock_U(128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)

        return x_out


class DecoderBlockLinkNet(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C/4, 2 * H, 2 * W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                          stride=2, padding=1, output_padding=0)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x


class LinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            x_out = F.log_softmax(f5, dim=1)
        else:
            x_out = f5
        return x_out


class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int, bn=False):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


class UNetModule(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(in_, out)
        self.l2 = Conv3BN(out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class UNet(nn.Module):
    """
    Vanilla UNet.

    Implementation from https://github.com/lopuhin/mapillary-vistas-2017/blob/master/unet_models.py
    """
    output_downscaled = 1
    module = UNetModule

    def __init__(self,
                 input_channels: int = 3,
                 filters_base: int = 32,
                 down_filter_factors=(1, 2, 4, 8, 16),
                 up_filter_factors=(1, 2, 4, 8, 16),
                 bottom_s=4,
                 num_classes=1,
                 add_output=True):
        super().__init__()
        self.num_classes = num_classes
        assert len(down_filter_factors) == len(up_filter_factors)
        assert down_filter_factors[-1] == up_filter_factors[-1]
        down_filter_sizes = [filters_base * s for s in down_filter_factors]
        up_filter_sizes = [filters_base * s for s in up_filter_factors]
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        self.down.append(self.module(input_channels, down_filter_sizes[0]))
        for prev_i, nf in enumerate(down_filter_sizes[1:]):
            self.down.append(self.module(down_filter_sizes[prev_i], nf))
        for prev_i, nf in enumerate(up_filter_sizes[1:]):
            self.up.append(self.module(
                down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i]))
        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample = nn.Upsample(scale_factor=2)
        upsample_bottom = nn.Upsample(scale_factor=bottom_s)
        self.downsamplers = [None] + [pool] * (len(self.down) - 1)
        self.downsamplers[-1] = pool_bottom
        self.upsamplers = [upsample] * len(self.up)
        self.upsamplers[-1] = upsample_bottom
        self.add_output = add_output
        if add_output:
            self.conv_final = nn.Conv2d(up_filter_sizes[0], num_classes, 1)

    def forward(self, x):
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)

        x_out = xs[-1]
        for x_skip, upsample, up in reversed(
                list(zip(xs[:-1], self.upsamplers, self.up))):
            x_out = upsample(x_out)
            x_out = up(torch.cat([x_out, x_skip], 1))

        if self.add_output:
            x_out = self.conv_final(x_out)
            if self.num_classes > 1:
                x_out = F.log_softmax(x_out, dim=1)
        return x_out


class AlbuNet(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
        """

    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.resnet34(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlock(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlock(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlock(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)

        return x_out
