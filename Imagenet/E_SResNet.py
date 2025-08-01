import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from spikingjelly.clock_driven import layer, surrogate
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode, MultiStepIFNode
from src.anti_aliasing import AntiAliasDownsampleLayer
from src.avg_pool import FastAvgPool2d
from src.general_layers import SEModule, SpaceToDepthModule

import torchvision

class ReflectPadNoGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pad):
        ctx.save_for_backward(torch.tensor([x.size(2), x.size(3)]))
        ctx.pad = pad
        return F.pad(x, pad=pad, mode='reflect')

    @staticmethod
    def backward(ctx, grad_output):
        orig_h, orig_w = ctx.saved_tensors[0].int().tolist()
        pad = ctx.pad
        if len(pad) == 4:
            left, right, top, bottom = pad
        elif len(pad) == 2:
            left = right = pad[0]
            top = bottom = pad[1]
        else:
            raise ValueError("Invalid padding format")
        grad_input = grad_output.narrow(2, top, orig_h).narrow(3, left, orig_w)
        return grad_input, None

class StableReflectPad(nn.Module):
    def __init__(self, padding=(1, 1, 1, 1)):
        super().__init__()
        self.padding = padding
    def forward(self, x):
        return ReflectPadNoGrad.apply(x, self.padding)

surrogate_function = surrogate.Sigmoid()


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, use_se=True, anti_alias_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('SpikingBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in SpikingBasicBlock")

        if stride == 1:
            self.conv1 = layer.SeqToANNContainer(
                conv3x3(inplanes, planes, stride),
                norm_layer(planes)
            )
        else:
            if anti_alias_layer is None:
                self.conv1 = layer.SeqToANNContainer(
                    conv3x3(inplanes, planes, stride=2),
                    norm_layer(planes)
                )
            else:
                self.conv1 = layer.SeqToANNContainer(
                    conv3x3(inplanes, planes, stride=1),
                    norm_layer(planes)
                )

        self.sn1 = MultiStepIFNode(detach_reset=True, surrogate_function=surrogate_function, backend='cupy')

        self.conv2 = layer.SeqToANNContainer(
            conv3x3(planes, planes),
            norm_layer(planes)
        )

        if stride == 1:
            self.aa = None
        else:
            if anti_alias_layer is None:
                self.aa = None
            else:
                self.pad = layer.SeqToANNContainer(
                    StableReflectPad((1, 1, 1, 1))
                )
                self.sn = MultiStepIFNode(detach_reset=True, surrogate_function=surrogate_function, backend='cupy')
                self.aa = layer.SeqToANNContainer(
                    anti_alias_layer(channels=planes, filt_size=3, stride=2)
                )

        self.downsample = downsample
        self.stride = stride
        self.sn2 = MultiStepIFNode(detach_reset=True, surrogate_function=surrogate_function, backend='cupy')
        reduce_layer_planes = planes * self.expansion // 32
        self.se = SEModule(planes * self.expansion, reduce_layer_planes) if use_se else None

    def forward(self, x):
        identity = x
        out = self.conv1(self.sn1(x))
        out = self.conv2(self.sn2(out))
        if self.aa is not None:
            out = self.pad(out)
            out = self.aa(self.sn(out))

        if self.se is not None: out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, use_se=True, anti_alias_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = layer.SeqToANNContainer(
            conv1x1(inplanes, width),
            norm_layer(width)
        )
        self.sn1 = MultiStepIFNode(detach_reset=True, surrogate_function=surrogate_function, backend='cupy')

        self.conv2 = layer.SeqToANNContainer(
            conv3x3(width, width, stride, groups, dilation),
            norm_layer(width)
        )
        self.sn2 = MultiStepIFNode(detach_reset=True, surrogate_function=surrogate_function, backend='cupy')

        self.conv3 = layer.SeqToANNContainer(
            conv1x1(width, planes * self.expansion),
            norm_layer(planes * self.expansion)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn3 = MultiStepIFNode(detach_reset=True, surrogate_function=surrogate_function, backend='cupy')
        reduce_layer_planes = planes * self.expansion // 32
        self.se = SEModule(planes, reduce_layer_planes) if use_se else None


    def forward(self, x):
        identity = x

        out = self.conv1(self.sn1(x))

        out = self.conv2(self.sn2(out))

        if self.se is not None: out = self.se(out)

        out = self.conv3(self.sn3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out


class ESResNet(nn.Module):

    def __init__(self, layers, in_chans=3, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, T=4, remove_model_jit=False):
        super(ESResNet, self).__init__()

        self.space_to_depth = SpaceToDepthModule(remove_model_jit=remove_model_jit)
        anti_alias_layer = partial(AntiAliasDownsampleLayer, remove_aa_jit=remove_model_jit)

        self.T = T
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.planes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_chans * 16, self.planes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.planes)

        self.sn1 = MultiStepIFNode(detach_reset=True, surrogate_function=surrogate_function, backend='cupy')
        self.maxpool = layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(BasicBlock, 64, layers[0], use_se=True,
                                       anti_alias_layer=anti_alias_layer)
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], use_se=True, anti_alias_layer=anti_alias_layer)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], use_se=True, anti_alias_layer=anti_alias_layer)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], use_se=False, anti_alias_layer=anti_alias_layer)
        self.avgpool = layer.SeqToANNContainer(FastAvgPool2d())
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, use_se=True, anti_alias_layer=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                MultiStepIFNode(detach_reset=True, surrogate_function=surrogate_function, backend='cupy'),
                layer.SeqToANNContainer(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, use_se=use_se, anti_alias_layer=anti_alias_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, use_se=use_se, anti_alias_layer=anti_alias_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, return_feature=False):
        x = self.space_to_depth(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x.unsqueeze_(0)
        x = x.repeat(self.T, 1, 1, 1, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.sn1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.fc(x)
        return x.mean(dim=0)

    def forward(self, x, return_feature=False):
        return self._forward_impl(x, return_feature)


def E_SResNet(layers, **kwargs):
    model = ESResNet(layers, **kwargs)
    return model


def E_SResNet_S(**kwargs):
    return E_SResNet([2, 2, 2, 2], **kwargs)


def E_SResNet_M(**kwargs):
    return E_SResNet([3, 4, 6, 3], **kwargs)

