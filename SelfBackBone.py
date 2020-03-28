import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ASFF(nn.Module):

    def __init__(self,numberOfFeatures,outTensorIndex):
        super(ASFF,self).__init__()
        self.numFea = numberOfFeatures
        self.index = outTensorIndex
        self.lambdas = nn.Parameter(torch.zeros(size=[numberOfFeatures],requires_grad=True).float(),requires_grad=True)

    def forward(self,x) :
        """
        :param x: List[Tensors]
        :return:
        """
        outSize = x[self.index].shape[2:]
        weights = torch.softmax(self.lambdas,dim=0)
        weightsList = torch.chunk(weights,self.numFea,dim=0)
        feature = F.interpolate(x[0], size=outSize, mode="bilinear", align_corners=True) * weightsList[0]
        for i in range(1,self.numFea):
            feature = feature + F.interpolate(x[i], size=outSize, mode="bilinear", align_corners=True) * weightsList[i]
        return feature



class ResNet(nn.Module):

    def __init__(self,  layers, block = Bottleneck, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.GroupNorm(4,self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.Conv2d(self.inplanes,64,3,2,1,groups=4)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.convL2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.convL3 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.convL4 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(p = 0.1)
        self.asff2 = ASFF(3,0)
        self.asff3 = ASFF(3,1)
        self.asff4 = ASFF(3,2)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x1 = self.dropout(x1)
        #print(out1.shape)
        x2 = self.layer2(x1)
        x2 = self.dropout(x2)
        out2 = self.convL2(x2)
        #print(out2.shape)
        x3 = self.layer3(x2)
        x3 = self.dropout(x3)
        out3 = self.convL3(x3)
        #print(out3.shape)
        x4 = self.layer4(x3)
        x4 = self.dropout(x4)
        out4 = self.convL4(x4)
        #print(out4.shape)
        #print(out3.shape[2:])
        out4_2x = F.interpolate(out4,size=out3.shape[2:],mode="bilinear",align_corners=True)
        finalOut3 = torch.add(out4_2x,out3)
        out3_2x = F.interpolate(finalOut3,size=out2.shape[2:],mode="bilinear",align_corners=True)
        finalOut2 = torch.add(out3_2x,out2)

        asff4 = self.asff4([out4,finalOut3,finalOut2])
        asff3 = self.asff3([out4, finalOut3, finalOut2])
        asff2 = self.asff2([out4, finalOut3, finalOut2])

        return OrderedDict([('0', asff4),("1",asff3),("2",asff2)])

if __name__ == "__main__":
    testResNet = ResNet(layers=[3, 4, 16, 3])
    testInput = torch.randn(size=[5,3,64,64])
    print(testResNet(testInput)["0"].shape)
    print(testResNet(testInput)["1"].shape)
    print(testResNet(testInput)["2"].shape)