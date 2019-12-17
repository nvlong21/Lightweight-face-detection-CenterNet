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
class IDAUp(nn.Module):
    def __init__(self, out_dim, channel):
        super(IDAUp, self).__init__()
        self.out_dim = out_dim
        self.up = nn.Sequential(
                    nn.ConvTranspose2d(
                        out_dim, out_dim, kernel_size=2, stride=2, padding=0,
                        output_padding=0, groups=out_dim, bias=False),
                    nn.BatchNorm2d(out_dim,eps=0.001,momentum=0.1),
                    nn.ReLU())
        self.conv =  nn.Sequential(
                    nn.Conv2d(channel, out_dim,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_dim,eps=0.001,momentum=0.1),
                    nn.ReLU(inplace=True))

    def forward(self, inpu1, input2):
        x = self.up(inpu1)
        y = self.conv(input2)
        out = x + y
        return out
class ShuffleNetV2(nn.Module):
    def __init__(self, input_size=224, model_size='0.5x'):
        super(ShuffleNetV2, self).__init__()
        print('model size is ', model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192,1024]
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976, 1024]
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
        self.conv_last  = conv_1x1_bn(input_channel, 24)
        self.up1 = IDAUp(24, 96)
        self.up2 = IDAUp(24, 48)
        self.up3 = IDAUp(24, 24)

        self.heads = {
                       'hm': 1,
                       'wh': 2, 
                       'lm': 10,
                       'reg': 2
                       }

        for head in self.heads:
            out_c = self.heads[head]
            fc = nn.Sequential(
                  nn.Conv2d(24, 24,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(24, out_c, 
                    kernel_size=1, stride=1, 
                    padding=0, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.first_conv(x)
        x_max = self.maxpool(x)
        x = self.shuff_1(x_max)
        x = self.shuff_2(x)
        x = self.shuff_3(x)
        x4 = self.shuff_4(x)
        x = self.shuff_5(x4)
        x = self.shuff_6(x)
        x = self.shuff_7(x)
        x = self.shuff_8(x)
        
        x = self.shuff_9(x)
        x = self.shuff_10(x)
        x = self.shuff_11(x)
        x12 = self.shuff_12(x)
        x = self.shuff_13(x12)

        x = self.shuff_14(x)
        x = self.shuff_15(x)
        x= self.shuff_16(x)
        x = self.conv_last(x)

        x = self.up1(x, x12)
        x = self.up2(x, x4)
        x = self.up3(x, x_max)

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(x)
            if 'hm' in head and not self.training:
                z[head] = F.sigmoid(z[head])
        return [z]


import time
from torchsummary import summary
if __name__ == "__main__":
    
    model = ShuffleNetV2()#.cuda()
    model.eval()
    test_data = torch.rand(1, 3, 640, 640)#.cuda()
    # summary(model, (3,640, 640))
    for i in range(15):
        t = time.time()
        test_outputs = model(test_data) #, test_data_2]
        t2 = time.time()
        print(t2 -t)
        t = t2
