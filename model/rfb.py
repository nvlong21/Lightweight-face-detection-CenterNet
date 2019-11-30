import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class DUC(nn.Module):
    '''
    Initialize: inplanes, planes, upscale_factor
    OUTPUT: (planes // upscale_factor^2) * ht * wd
    '''
    def __init__(self, inplanes, planes, kernel_size = 3, upscale_factor=2):
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(
            inplanes, planes, kernel_size=kernel_size, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        # fill_up_weights(self.pixel_shuffle)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x
def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )
class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        if bn:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            self.relu = nn.ReLU(inplace=True) if relu else None
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
            self.bn = None
            self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 1, dilation=vision + 1, relu=False, groups=groups)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2, dilation=vision + 2, relu=False, groups=groups)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, groups=groups),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4, dilation=vision + 4, relu=False, groups=groups)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
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


class Mb_Tiny_RFB(nn.Module):

    def __init__(self):
        super(Mb_Tiny_RFB, self).__init__()
        self.base_channel = 8 * 2

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, self.base_channel, 2),  # 160*120
            conv_dw(self.base_channel, self.base_channel * 2, 1),
            conv_dw(self.base_channel * 2, self.base_channel * 2, 2),  # 80*60
            conv_dw(self.base_channel * 2, self.base_channel * 2, 1),
            conv_dw(self.base_channel * 2, self.base_channel * 4, 2),  # 40*30
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            BasicRFB(self.base_channel * 4, self.base_channel * 4, stride=1, scale=1.0),
            conv_dw(self.base_channel * 4, self.base_channel * 8, 2),  # 20*15
            conv_dw(self.base_channel * 8, self.base_channel * 8, 1),
            conv_dw(self.base_channel * 8, self.base_channel * 8, 1),
            conv_dw(self.base_channel * 8, self.base_channel * 16, 2),  # 10*8
            conv_dw(self.base_channel * 16, self.base_channel * 16, 1)
        )
        # self.conv_compress = nn.Conv2d(256, 512, kernel_size = 1, stride = 1, padding=0, bias=False)

        self.duc = nn.Sequential(DUC(256, 512, kernel_size = 3, upscale_factor=2), 
                                DUC(128, 256, kernel_size = 3, upscale_factor=2),
                                DUC(64, 128, kernel_size = 3, upscale_factor=2))
        self.heads = {
                       'hm': 1,
                       'wh': 2, 
                       'lm': 10,
                       'reg': 2
                       }

        for head in self.heads:
            out_c = self.heads[head]
            fc = nn.Sequential(
                 nn.Conv2d(32, 64, 3, 1, 1,  bias=False),
                 nn.BatchNorm2d(64),
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
        x = self.model(x)
        x = self.duc(x)
        
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(x)
            if 'hm' in head and not self.training:
                z[head] = F.sigmoid(z[head])
        return [z]

        return x
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
if __name__ == '__main__':
    net = Mb_Tiny_RFB().cuda()
    net.eval()
    input = torch.FloatTensor(1, 3, 640, 640).uniform_(0, 1).cuda()
    a = torch.rand(1, 3, 2048, 2048)#.cuda()
    summary(net, (3, 640, 640))
    for i in range(5):
        t = time.time()
        iop = net(input)
        t2 = time.time()
        print( t2 -t)