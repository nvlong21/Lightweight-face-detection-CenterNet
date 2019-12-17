import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math

import torchsummary

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

class Context(nn.Module):
    def __init__(self,inchannels=64):
        super(Context,self).__init__()
        self.context_plain = inchannels//2
        self.conv1 = CB(inchannels)
        self.conv2 = CBR(inchannels,self.context_plain)
        self.conv2_1 = CB(self.context_plain)
        self.conv2_2_1 = CBR(self.context_plain,self.context_plain)
        self.conv2_2_2 = CB(self.context_plain)
        self.concat = Concat()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self,x):
        f1 = self.conv1(x)
        f2_ = self.conv2(x)
        f2 = self.conv2_1(f2_)
        f3 = self.conv2_2_1(f2_)
        f3 = self.conv2_2_2(f3)

        #out = torch.cat([f1,f2,f3],dim=1)
        out = self.concat(f1,f2,f3)
        out = self.relu(out)

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
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )
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
class CenterFace_MobileNet(nn.Module):
    def __init__(self):
        super(CenterFace_MobileNet, self).__init__()
        self.mobilenet0_conv0 = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(num_features=8, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv1 = nn.Sequential(
                    nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1, groups=8, bias=False),
                    nn.BatchNorm2d(num_features=8, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv2 = nn.Sequential(
                    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=16, momentum=0.9),
                    nn.ReLU(inplace=True))
        
        self.mobilenet0_conv3 = nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1, groups=16, bias=False),
                    nn.BatchNorm2d(num_features=16, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv4 = nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=32, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv5 = nn.Sequential(
                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32, bias=False),
                    nn.BatchNorm2d(num_features=32, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv6 = nn.Sequential(
                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=32, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv7 = nn.Sequential(
                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, groups=32, bias=False),
                    nn.BatchNorm2d(num_features=32, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv8 = nn.Sequential(
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=64, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv9 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=64, bias=False),
                    nn.BatchNorm2d(num_features=64, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv10 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=64, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv11 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, groups=64, bias=False),
                    nn.BatchNorm2d(num_features=64, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv12 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=128, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv13 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
                    nn.BatchNorm2d(num_features=128, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv14 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=128, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv15 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
                    nn.BatchNorm2d(num_features=128),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv16 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=128, momentum=0.9),
                    nn.ReLU(inplace=True))
        
        self.mobilenet0_conv17 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
                    nn.BatchNorm2d(num_features=128, momentum=0.9),
                    nn.ReLU(inplace=True))
        
        self.mobilenet0_conv18 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=128, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv19 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
                    nn.BatchNorm2d(num_features=128, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv20 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=128, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv21 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
                    nn.BatchNorm2d(num_features=128, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv22 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=128, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv23 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, groups=128, bias=False),
                    nn.BatchNorm2d(num_features=128, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv24 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=256, momentum=0.9),
                    nn.ReLU(inplace=True))
        
        self.mobilenet0_conv25 = nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=256, bias=False),
                    nn.BatchNorm2d(num_features=256, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.mobilenet0_conv26 = nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=256, momentum=0.9),
                    nn.ReLU(inplace=True))
        self.up1 = IDAUp(24, 128)
        self.up2 = IDAUp(24, 64)
        self.up3 = IDAUp(24, 32)
        self.conv_last = conv_1x1_bn(256, 24)
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
                    kernel_size=3, padding=1, bias=False),
                  nn.BatchNorm2d(24,eps=1e-5,momentum=0.01),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(24, out_c, 
                    kernel_size=1, stride=1, 
                    padding=0, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

        self.freeze_bn()
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, x):
        x = self.mobilenet0_conv0(x)
        x = self.mobilenet0_conv1(x)

        x = self.mobilenet0_conv2(x)
        x = self.mobilenet0_conv3(x)
        x = self.mobilenet0_conv4(x)
        x = self.mobilenet0_conv5(x)
        x6 = self.mobilenet0_conv6(x)
        x = self.mobilenet0_conv7(x6)

        x = self.mobilenet0_conv8(x)
        x = self.mobilenet0_conv9(x)
        x10 = self.mobilenet0_conv10(x)
        x = self.mobilenet0_conv11(x10)
        x = self.mobilenet0_conv12(x)
        x = self.mobilenet0_conv13(x)
        x = self.mobilenet0_conv14(x)
        x = self.mobilenet0_conv15(x)
        x = self.mobilenet0_conv16(x)
        x = self.mobilenet0_conv17(x)
        x = self.mobilenet0_conv18(x)
        x = self.mobilenet0_conv19(x)
        x = self.mobilenet0_conv20(x)
        x = self.mobilenet0_conv21(x)
        x22 = self.mobilenet0_conv22(x)
        x = self.mobilenet0_conv23(x22)
        x = self.mobilenet0_conv24(x)
        x = self.mobilenet0_conv25(x)
        x = self.mobilenet0_conv26(x)
        x = self.conv_last(x)
        x = self.up1(x, x22)
        x = self.up2(x, x10)
        x = self.up3(x, x6)

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(x)
            if 'hm' in head and not self.training:
                z[head] = F.sigmoid(z[head])
        # print(z['hm'])
        return [z]



import time
from torchsummary import summary
if __name__ == '__main__':
    net = CenterFace_MobileNet().cuda()
    input = torch.FloatTensor(1, 3, 640, 640).uniform_(0, 1).cuda()
    summary(net, (3,640, 640))
    for i in range(5):
        t = time.time()
        # print(net(input))
        t2 = time.time()
        print( t2 -t)
