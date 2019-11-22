import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
try:
    from .layer import BasicBlock, Bottleneck, RegressionTransform
    from .anchors import Anchors
    from . import losses
    from .blocks import ShuffleV2Block
except:
    from layer import BasicBlock, Bottleneck, RegressionTransform
    from anchors import Anchors
    import losses
    from blocks import ShuffleV2Block
import torchsummary
class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=128):
        super(PyramidFeatures, self).__init__()
        
        # upsample C5 to get P5 from the FPN paper
        self.P5_1           = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1           = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1           = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        # self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):

        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)
        
        P4_x = self.P4_1(C4)
        # P4_x = F.interpolate(P4_x, size=[P5_upsampled_x.size(2), P5_upsampled_x.size(3)])
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        # P3_x = F.interpolate(P3_x, size=[P4_upsampled_x.size(2), P4_upsampled_x.size(3)])
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]

class ClassHead(nn.Module):
    def __init__(self,inchannels=64,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1)

        # if use focal loss instead of OHEM
        #self.output_act = nn.Sigmoid()

        # if use OHEM
        self.output_act = nn.Softmax(dim=-1)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1)
        b, h, w, c = out.shape
        out = out.view(b, h, w, self.num_anchors, 2)
        #out = out.permute(0,2,3,1).contiguous().view(out.shape[0], -1, 2)
        out = self.output_act(out)
        
        return out.contiguous().view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=64,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1)

        return out.contiguous().view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=64,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1)

        return out.contiguous().view(out.shape[0], -1, 10)

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

class RetinaFace_MobileNet(nn.Module):
    def __init__(self):
        super(RetinaFace_MobileNet, self).__init__()
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

        self.fpn = PyramidFeatures(64, 128, 256)

        self.context = Context(inchannels=128)       
        
        self.clsHead = ClassHead(inchannels = 256)
        self.bboxHead = BboxHead(inchannels=  256)
        self.ldmHead = LandmarkHead(inchannels = 256)

        self.anchors = Anchors()

        self.regressBoxes = RegressionTransform()
        
        self.losslayer = losses.LossLayer()

        self.freeze_bn()
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, x):
        if self.training:
            img_batch, annotations = x
        else:
            img_batch = inputs
#        batchsize = img_batch.shape[0]
        x = self.mobilenet0_conv0(img_batch)
        x = self.mobilenet0_conv1(x)
        x = self.mobilenet0_conv2(x)
        x = self.mobilenet0_conv3(x)
        x = self.mobilenet0_conv4(x)
        x = self.mobilenet0_conv5(x)
        x = self.mobilenet0_conv6(x)
        x = self.mobilenet0_conv7(x)
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
        x26 = self.mobilenet0_conv26(x)
        features = self.fpn([x10, x22, x26])
        context_features = [self.context(feature) for i,feature in enumerate(features)]

        bbox_regressions = torch.cat([self.bboxHead(feature) for i,feature in enumerate(context_features)], dim=1)
        ldm_regressions = torch.cat([self.ldmHead(feature) for i,feature in enumerate(context_features)], dim=1)
        classifications = torch.cat([self.clsHead(feature) for i,feature in enumerate(context_features)], dim=1)

        # bbox_regressions = torch.cat([self.bboxHead(feature) for feature in features], dim=1)
        # ldm_regressions = torch.cat([self.ldmHead(feature) for feature in features], dim=1)
        # classifications = torch.cat([self.clsHead(feature) for feature in features],dim=1)

        anchors = self.anchors(img_batch)

        if self.training:
            return self.losslayer(classifications, bbox_regressions,ldm_regressions, anchors, annotations)
        else:
            bboxes, landmarks = self.regressBoxes(anchors, bbox_regressions, ldm_regressions, img_batch)

            return classifications, bboxes, landmarks

def load_retinaface_mbnet(path = ''):
    net = RetinaFace_MobileNet()
    ctx = mx.cpu()
    sym, arg_params, aux_params = mx.model.load_checkpoint(path, 0)
    args = arg_params.keys()
    auxs = aux_params.keys()
    weights = []

    layers = sym.get_internals().list_outputs()
    for layer in layers:
        for arg in args:
            if layer == arg:
                weights.append(arg_params[arg].asnumpy())

        for aux in auxs:
            if layer == aux:
                weights.append(aux_params[aux].asnumpy())

    net_dict = net.state_dict()
    net_layers = list(net_dict.keys())
    idx = 0
    for layer in net_layers:
        if 'num_batches_tracked' not in layer:
            net_dict[layer] = torch.from_numpy(weights[idx]).type(torch.FloatTensor)
            idx += 1
    net.load_state_dict(net_dict)
    # model_dict = {
    # 'state_dict': net_dict
    # }
    # torch.save(model_dict, "checkpoint.pt")
    net.eval()
    return net
import time
if __name__ == '__main__':
    net = RetinaFace_MobileNet().cuda()
    input = torch.FloatTensor(1, 3, 640, 640).uniform_(0, 1).cuda()
    anno = torch.FloatTensor(1, 1, 5).uniform_(0, 1).cuda()
    for i in range(5):
        t = time.time()
        print(net([input, anno]))
        t2 = time.time()
        print( t2 -t)
