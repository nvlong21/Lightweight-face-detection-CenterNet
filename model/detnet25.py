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
# import dsntnn
BN_MOMENTUM = 0.1

class DUC(nn.Module):
    '''
    Initialize: inplanes, planes, upscale_factor
    OUTPUT: (planes // upscale_factor^2) * ht * wd
    '''

    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(
            inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x

class CBR(nn.Module):
    def __init__(self,inchannels,outchannels):
        super(CBR,self).__init__()
        self.conv3x3 = nn.Conv2d(inchannels,outchannels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.normal_(m.weight, std=0.01)      

    def forward(self,x):
        x = self.conv3x3(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class CB(nn.Module):
    def __init__(self,inchannels):
        super(CB,self).__init__()
        self.conv3x3 = nn.Conv2d(inchannels,inchannels, kernel_size=3,stride=1,padding=1,bias=False)
        self.bn = nn.BatchNorm2d(inchannels)
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.normal_(m.weight, std=0.01)

    def forward(self,x):
        x = self.conv3x3(x)
        x = self.bn(x)

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
def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1, stride=1, relu=True, bias=True):
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)
class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class UShapedContextBlock(nn.Module):
    def __init__(self, in_channels, to_onnx=False):
        super().__init__()
        self.to_onnx = to_onnx
        self.encoder1 = nn.Sequential(
            conv(in_channels, in_channels*2, stride=2),
            conv(in_channels*2, in_channels*2),
        )
        self.encoder2 = nn.Sequential(
            conv(in_channels*2, in_channels*2, stride=2),
            conv(in_channels*2, in_channels*2),
        )
        self.decoder2 = nn.Sequential(
            conv(in_channels*2 + in_channels*2, in_channels*2),
            conv(in_channels*2, in_channels*2),
        )
        self.decoder1 = nn.Sequential(
            conv(in_channels*3, in_channels*2),
            conv(in_channels*2, in_channels)
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)

        size_e1 = (e1.size()[2], e1.size()[3])
        size_x = (x.size()[2], x.size()[3])
        if self.to_onnx:  # Need interpolation to fixed size for conversion
            size_e1 = (16, 16)
            size_x = (32, 32)
        d2 = self.decoder2(torch.cat([e1, F.interpolate(e2, size=size_e1,
                                                        mode='bilinear', align_corners=False)], 1))
        d1 = self.decoder1(torch.cat([x, F.interpolate(d2, size=size_x,
                                                       mode='bilinear', align_corners=False)], 1))
        return d1
class ShuffleNetV2(nn.Module):
    def __init__(self, input_size=224, n_class=1000, model_size='1.5x'):
        super(ShuffleNetV2, self).__init__()
        print('model size is ', model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192, 384]
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
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
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
        self.conv_last      = conv_1x1_bn(input_channel, self.stage_out_channels[-1])
        self.conv_compress = nn.Conv2d(self.stage_out_channels[-1], 512, kernel_size = 1, stride = 1, padding=0, bias=False)
        # self.trunk = nn.Sequential(
        #     UShapedContextBlock(464, False),
        #     RefinementStageBlock(464, 256),
        #     RefinementStageBlock(256, 256),
        #     RefinementStageBlock(256, 64),
        # )

        self.duc1 = DUC(512, 1024, upscale_factor=2)
        self.duc2 = DUC(256, 512, upscale_factor=2)
        self.duc3 = DUC(128, 256, upscale_factor=2)

        self.heads = {
                       'hm': 1,
                       'wh': 2, 
                       'lm': 10,
                       'reg': 2
                       }

        for head in self.heads:
            out_c = self.heads[head]
            fc = nn.Sequential(
                  nn.Conv2d(64, 128,
                    kernel_size=1, padding=0, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(128, out_c, 
                    kernel_size=1, stride=1, 
                    padding=0, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
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
        x = self.conv_compress(x)
        # x = self.trunk(x)
        # print(x.size())
        # 
        x = self.duc1(x)
        x = self.duc2(x)
        x = self.duc3(x)

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(x)
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
if __name__ == "__main__":
    
    model = ShuffleNetV2().cuda()
    model.eval()
    test_data = torch.rand(1, 3, 124, 153).cuda()
    test_data_2 = torch.rand(5, 1, 5)#.cuda()
    for i in range(15):
        t = time.time()
        test_outputs = model(test_data) #, test_data_2]
        t2 = time.time()
        print(t2 -t)
        t = t2
