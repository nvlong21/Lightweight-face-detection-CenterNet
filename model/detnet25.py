import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
try:
    from .blocks import ShuffleV2Block
except:
    from blocks import ShuffleV2Block
import torchsummary
from collections import OrderedDict
# import dsntnn
BN_MOMENTUM = 0.1
class GCN(nn.Module):
    def __init__(self, inplanes, planes, ks=3):
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(inplanes, planes, kernel_size=(ks, 1),
                                 padding=(ks//2, 0))

        self.conv_l2 = nn.Conv2d(planes, planes, kernel_size=(1, ks),
                                 padding=(0, ks//2))
        self.conv_r1 = nn.Conv2d(inplanes, planes, kernel_size=(1, ks),
                                 padding=(0, ks//2))
        self.conv_r2 = nn.Conv2d(planes, planes, kernel_size=(ks, 1),
                                 padding=(ks//2, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)

        x = x_l + x_r

        return x
class DUC(nn.Module):
    '''
    Initialize: inplanes, planes, upscale_factor
    OUTPUT: (planes // upscale_factor^2) * ht * wd
    '''
    def __init__(self, inplanes, planes, upscale_factor=2, kernel_size = 3):
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(
            inplanes, planes, kernel_size=kernel_size, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x

class Concat(nn.Module):
    def forward(self,*feature):
        out = torch.cat(feature,dim=1)
        return out


def initialize_layer(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.normal_(layer.weight, std=0.01)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, val=0)

        for layer in self.context:
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes//2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)
    
    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

class ASPPInPlaceABNBlock(nn.Module):
    def __init__(self, in_chs, out_chs, feat_res=(56, 112),
                 up_ratio=2, aspp_sec=(12, 24, 36)):
        super(ASPPInPlaceABNBlock, self).__init__()

        self.in_norm = IBN(in_chs)
        self.gave_pool = nn.Sequential(OrderedDict([("conv1_0", nn.Conv2d(in_chs, out_chs,
                                                                          kernel_size=1, stride=1, padding=0,
                                                                          groups=1, bias=False, dilation=1)),
                                                    ("up0", nn.Upsample(scale_factor = 2, mode='nearest'))]))


        self.aspp_bra = nn.Sequential(OrderedDict([("conv2_3", nn.Conv2d(in_chs, out_chs, kernel_size=3,
                                                                          stride=1, padding=1, bias=False,
                                                                          groups=1))]))
        self.aspp_bra1 = nn.Sequential(OrderedDict([("conv2_4", nn.Conv2d(in_chs, out_chs, kernel_size=3,
                                                                          stride=1, padding=1, bias=False,
                                                                          groups=1))]))

        self.aspp_catdown = nn.Sequential(OrderedDict([("norm_act", IBN(2*out_chs)),
                                                       ("conv_down", nn.Conv2d(2*out_chs, out_chs, kernel_size=1,
                                                                               stride=1, padding=1, bias=False,
                                                                               groups=1, dilation=1)),
                                                       ("dropout", nn.Dropout2d(p=0.2, inplace=True))]))

        self.upsampling = nn.Upsample(size=(int(feat_res[0]*up_ratio), int(feat_res[1]*up_ratio)), mode='bilinear')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # channel_shuffle: shuffle channels in groups
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    @staticmethod
    def _channel_shuffle(x, groups):
        """
        Channel shuffle operation
        :param x: input tensor
        :param groups: split channels into groups
        :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()

        channels_per_group = num_channels // groups

        # reshape
        x = x.view(batch_size, groups, channels_per_group, height, width)

        # transpose
        # - contiguous() required if transpose() is used before view().
        #   See https://github.com/pytorch/pytorch/issues/764
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, height, width)

        return x

    def forward(self, x):
        x = self.in_norm(x)
        x1= self.gave_pool(x)
        x2 = self.aspp_bra(x)
        x3 = self.aspp_bra1(x)
        x = torch.cat([x2, x3], dim=1)
        return x

        # out = self.aspp_catdown(x)
        # return out, self.upsampling(out)
class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)
class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput, k_size = 3, stride=2, padding=1, output_padding=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, k_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class Decoder (nn.Module):
    def __init__(self,  ninput):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(UpsamplerBlock(ninput, ninput//2))
        self.layers.append(nn.Conv2d(ninput//2, ninput//2, 1, stride=1))
        self.layers.append(nn.BatchNorm2d(ninput//2))

        self.layers.append(UpsamplerBlock(ninput//2, ninput//4))
        self.layers.append(nn.Conv2d(ninput//4, ninput//4, 1, stride=1))
        self.layers.append(nn.BatchNorm2d(ninput//4))

        self.output_conv = nn.ConvTranspose2d(ninput//4, ninput//4, 1, stride=2, padding=0, output_padding=1, bias=True)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output
class ShuffleNetV2(nn.Module):
    def __init__(self, input_size=224, model_size='1.0x'):
        super(ShuffleNetV2, self).__init__()
        print('model size is ', model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192]
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976]
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel, eps=1e-3),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        output_channel = self.stage_out_channels[2]
        inp, outp, stride = input_channel, output_channel, 2
        self.shuff_1 = ShuffleV2Block(inp, outp, mid_channels=outp // 2, ksize=3, stride=stride)
        input_channel = output_channel
        inp, outp, stride = input_channel // 2, output_channel, 1
        self.shuff_2 = ShuffleV2Block(inp, outp, mid_channels=outp // 2, ksize=3, stride=stride)
        self.shuff_3 = ShuffleV2Block(inp, outp, mid_channels=outp // 2, ksize=3, stride=stride)
        self.shuff_4 = ShuffleV2Block(inp, outp, mid_channels=outp // 2, ksize=3, stride=stride)

        output_channel = self.stage_out_channels[3]
        inp, outp, stride = input_channel, output_channel, 2
        self.shuff_5 = ShuffleV2Block(inp, outp, mid_channels=outp // 2, ksize=3, stride=stride)
        input_channel = output_channel
        inp, outp, stride = input_channel // 2, output_channel, 1
        self.shuff_6 = ShuffleV2Block(inp, outp, mid_channels=outp // 2, ksize=3, stride=stride)
        self.shuff_7= ShuffleV2Block(inp, outp, mid_channels=outp // 2, ksize=3, stride=stride)

        self.shuff_8 = ShuffleV2Block(inp, outp, mid_channels=outp // 2, ksize=3, stride=stride)
        self.shuff_9 = ShuffleV2Block(inp, outp, mid_channels=outp // 2, ksize=3, stride=stride)

        self.shuff_10 = ShuffleV2Block(inp, outp, mid_channels=outp // 2, ksize=3, stride=stride)
        self.shuff_11 = ShuffleV2Block(inp, outp, mid_channels=outp // 2, ksize=3, stride=stride)
        self.shuff_12 = ShuffleV2Block(inp, outp, mid_channels=outp // 2, ksize=3, stride=stride)
        output_channel = self.stage_out_channels[4]
        inp, outp, stride = input_channel, output_channel, 2

        self.shuff_13 = ShuffleV2Block(inp, outp, mid_channels=outp // 2, ksize=3, stride=stride)
        input_channel = output_channel
        inp, outp, stride = input_channel//2, output_channel, 1

        self.shuff_14 = ShuffleV2Block(inp, outp, mid_channels=outp // 2, ksize=3, stride=stride)
        self.shuff_15 = ShuffleV2Block(inp, outp, mid_channels=outp // 2, ksize=3, stride=stride)

        self.shuff_16 = ShuffleV2Block(inp, outp, mid_channels=outp // 2, ksize=3, stride=stride)
        self.conv_last  = conv_1x1_bn(input_channel, 128)
        # self.decode = Decoder(256)
        self.decode = nn.Sequential(GCN(128, 256))#, GCN(128, 256), GCN(64,128)
        self.heads = {
                       'hm': 1,
                       'wh': 2, 
                       'lm': 10,
                       'reg': 2
                       }

        for head in self.heads:
            out_c = self.heads[head]
            fc = nn.Sequential(
                  nn.Conv2d(32, 64,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(64, out_c, 
                    kernel_size=1, stride=1, 
                    padding=0, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        t = time.time()
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.shuff_1(x)
        x = self.shuff_2(x)
        x = self.shuff_3(x)
        x = self.shuff_4(x)
        x = self.shuff_5(x)
        x = self.shuff_6(x)
        x = self.shuff_7(x)
        x = self.shuff_8(x)
        x = self.shuff_9(x)
        x = self.shuff_10(x)
        x = self.shuff_11(x)
        x = self.shuff_12(x)
        x = self.shuff_13(x)
        x = self.shuff_14(x)
        x = self.shuff_15(x)
        x= self.shuff_16(x)
        x = self.conv_last(x)
        
        # print("aaa:", time.time() -t)
        print(x.size())
        
        x = self.decode(x)
        print(x.size())

        # x = self.duc1(x)
        
        # x = self.duc2(x)
        # x = self.duc3(x)
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(x)
            if 'hm' in head and not self.training:
                z[head] = F.sigmoid(z[head])
        return [z]

        
    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

import time
from torchsummary import summary
if __name__ == "__main__":
    
    model = ShuffleNetV2().cuda()
    model.eval()
    test_data = torch.rand(1, 3, 640, 640).cuda()
    summary(model, (3,640, 640))
    for i in range(15):
        t = time.time()
        test_outputs = model(test_data) #, test_data_2]
        t2 = time.time()
        print(t2 -t)
        t = t2
