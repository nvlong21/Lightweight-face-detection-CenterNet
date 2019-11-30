import math

import mlconfig
import torch
from torch import nn
import torch.nn.functional as F
# from .utils import load_state_dict_from_url

model_urls = {
    'efficientnet_b0': 'https://www.dropbox.com/s/9wigibun8n260qm/efficientnet-b0-4cfa50.pth?dl=1',
    'efficientnet_b1': 'https://www.dropbox.com/s/6745ear79b1ltkh/efficientnet-b1-ef6aa7.pth?dl=1',
    'efficientnet_b2': 'https://www.dropbox.com/s/0dhtv1t5wkjg0iy/efficientnet-b2-7c98aa.pth?dl=1',
    'efficientnet_b3': 'https://www.dropbox.com/s/5uqok5gd33fom5p/efficientnet-b3-bdc7f4.pth?dl=1',
    'efficientnet_b4': 'https://www.dropbox.com/s/y2nqt750lixs8kc/efficientnet-b4-3e4967.pth?dl=1',
    'efficientnet_b5': 'https://www.dropbox.com/s/qxonlu3q02v9i47/efficientnet-b5-4c7978.pth?dl=1',
    'efficientnet_b6': None,
    'efficientnet_b7': None,
}

params = {
    'efficientnet_b0': (1.0, 1.0, 224, 0.2),
    'efficientnet_b1': (1.0, 1.1, 240, 0.2),
    'efficientnet_b2': (1.1, 1.2, 260, 0.3),
    'efficientnet_b3': (1.2, 1.4, 300, 0.3),
    'efficientnet_b4': (1.4, 1.8, 380, 0.4),
    'efficientnet_b5': (1.6, 2.2, 456, 0.4),
    'efficientnet_b6': (1.8, 2.6, 528, 0.5),
    'efficientnet_b7': (2.0, 3.1, 600, 0.5),
}


class Swish(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        padding = self._get_padding(kernel_size, stride)
        super(ConvBNReLU, self).__init__(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=0, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            Swish(),
        )

    def _get_padding(self, kernel_size, stride):
        p = max(kernel_size - stride, 0)
        return [p // 2, p - p // 2, p // 2, p - p // 2]


class SqueezeExcitation(nn.Module):

    def __init__(self, in_planes, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, reduced_dim, 1),
            Swish(),
            nn.Conv2d(reduced_dim, in_planes, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 expand_ratio,
                 kernel_size,
                 stride,
                 reduction_ratio=4,
                 drop_connect_rate=0.2, se = True):
        super(MBConvBlock, self).__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = in_planes == out_planes and stride == 1
        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))

        layers = []
        # pw
        if in_planes != hidden_dim:
            layers += [ConvBNReLU(in_planes, hidden_dim, 1)]
        layers += [
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim),
        ]
        if se:
            layers += [SqueezeExcitation(hidden_dim, reduced_dim),]
        # pw-linear
        layers += [
            nn.Conv2d(hidden_dim, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes)]

        self.conv = nn.Sequential(*layers)

    def _drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor

    def forward(self, x):
        if self.use_residual:
            return x + self._drop_connect(self.conv(x))
        else:
            return self.conv(x)


def _make_divisible(value, divisor=8):
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def _round_filters(filters, width_mult):
    if width_mult == 1.0:
        return filters
    return int(_make_divisible(filters * width_mult))


def _round_repeats(repeats, depth_mult):
    if depth_mult == 1.0:
        return repeats
    return int(math.ceil(depth_mult * repeats))
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

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)
class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput, k_size = 2, stride= 2, padding=0, output_padding=0):
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
        self.layers.append(non_bottleneck_1d(ninput//2, 0, 1))
        self.layers.append(non_bottleneck_1d(ninput//2, 0, 1))

        self.layers.append(UpsamplerBlock(ninput//2, ninput//4))
        self.layers.append(non_bottleneck_1d(ninput//4, 0, 1))
        self.layers.append(non_bottleneck_1d(ninput//4, 0, 1))

        self.output_conv = nn.ConvTranspose2d(ninput//4, ninput//4, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output
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
@mlconfig.register
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )
class EfficientNet(nn.Module):

    def __init__(self, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, num_classes=1000):
        super(EfficientNet, self).__init__()

        # yapf: disable
        settings = [
            # t,  c, n, s, k
            [1,  16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112
            [6,  24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56
            [6,  32, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28
            [6,  64, 2, 2, 3],  # MBConv6_3x3, SE,  28 ->  14
            [6, 96, 2, 1, 5],  # MBConv6_5x5, SE,  14 ->  14
            [6, 160, 2, 2, 5],  # MBConv6_5x5, SE,  14 ->   7
            [6, 320, 1, 1, 3]   # MBConv6_3x3, SE,   7 ->   7
        ]
        # yapf: enable

        out_channels = _round_filters(32, width_mult)
        features = [ConvBNReLU(3, out_channels, 3, stride=2)]
        in_channels = out_channels
        for t, c, n, s, k in settings:
            se = False
            out_channels = _round_filters(c, width_mult)
            repeats = _round_repeats(n, depth_mult)
            for i in range(repeats):
                stride = s if i == 0 else 1
                features += [MBConvBlock(in_channels, out_channels, expand_ratio=t, stride=stride, kernel_size=k, se= se)]
                in_channels = out_channels
        self.conv_last = conv_1x1_bn(320, 128)
        self.features = nn.Sequential(*features)
        self.decode = Decoder(128)
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
                    kernel_size=1, padding=0, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(64, out_c, 
                    kernel_size=1, stride=1, 
                    padding=0, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x  = self.conv_last(x)
        x = self.decode(x)
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(x)
            if 'hm' in head and not self.training:
                z[head] = F.sigmoid(z[head])
        return [z]


def _efficientnet(arch, pretrained, progress, **kwargs):
    width_mult, depth_mult, _, dropout_rate = params[arch]
    model = EfficientNet(width_mult, depth_mult, dropout_rate, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)

        if 'num_classes' in kwargs and kwargs['num_classes'] != 1000:
            del state_dict['classifier.1.weight']
            del state_dict['classifier.1.bias']

        model.load_state_dict(state_dict, strict=False)
    return model


@mlconfig.register
def efficientnet_b0(pretrained=False, progress=True, **kwargs):
    return _efficientnet('efficientnet_b0', pretrained, progress, **kwargs)


# @mlconfig.register
# def efficientnet_b1(pretrained=False, progress=True, **kwargs):
#     return _efficientnet('efficientnet_b1', pretrained, progress, **kwargs)


# @mlconfig.register
# def efficientnet_b2(pretrained=False, progress=True, **kwargs):
#     return _efficientnet('efficientnet_b2', pretrained, progress, **kwargs)


# @mlconfig.register
# def efficientnet_b3(pretrained=False, progress=True, **kwargs):
#     return _efficientnet('efficientnet_b3', pretrained, progress, **kwargs)


# @mlconfig.register
# def efficientnet_b4(pretrained=False, progress=True, **kwargs):
#     return _efficientnet('efficientnet_b4', pretrained, progress, **kwargs)


# @mlconfig.register
# def efficientnet_b5(pretrained=False, progress=True, **kwargs):
#     return _efficientnet('efficientnet_b5', pretrained, progress, **kwargs)


# @mlconfig.register
# def efficientnet_b6(pretrained=False, progress=True, **kwargs):
#     return _efficientnet('efficientnet_b6', pretrained, progress, **kwargs)


# @mlconfig.register
# def efficientnet_b7(pretrained=False, progress=True, **kwargs):
#     return _efficientnet('efficientnet_b7', pretrained, progress, **kwargs)
import time
from torchsummary import summary
if __name__ == "__main__":
    model = efficientnet_b0().cuda()
    
    # model = EfficientNet().cuda()
    model.eval()
    test_data = torch.rand(1, 3, 640, 640).cuda()
    summary(model, (3,640, 640))
    for i in range(5):
        t = time.time()
        test_outputs = model(test_data) #, test_data_2]
        t2 = time.time()
        print(t2 -t)
        t = t2