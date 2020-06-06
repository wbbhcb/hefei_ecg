# -*- coding: utf-8 -*-
'''
@time: 2019/10/11 9:42
直接修改torch的resnet
@ author: Chauncy
'''
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes,
                     kernel_size=7,
                     stride=stride,
                     padding=3,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes,
                               kernel_size=7,
                               bias=False,
                               padding=3)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes,
                               kernel_size=11,
                               stride=stride,
                               padding=5,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4,
                               kernel_size=7,
                               bias=False,
                               padding=3)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=34):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.globavepool = nn.AdaptiveAvgPool1d(1)
        self.globamaxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self.fc1 = nn.Linear(512,  256)
        self.fc_atten_1 = nn.Linear(512, 512)
        # self.fc_atten_2 = nn.Linear(128, 128)
        # self.fc_atten_3 = nn.Linear(256, 256)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def attention(self, x, i):
        out = torch.transpose(x, 1, 2)
        if i == 0:
            out = self.fc_atten_1(out)
        elif i == 1:
            out = self.fc_atten_2(out)
        elif i == 2:
            out = self.fc_atten_3(out)

        out = torch.transpose(out, 1, 2)
        weight = self.globavepool(out)
        out = weight * x
        return out

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # x = self.attention(x, 0)
        x = self.layer2(x)
        # x = self.attention(x, 1)
        x = self.layer3(x)
        # x = self.attention(x, 2)
        x = self.layer4(x)
        
        x = self.attention(x, 0)
        x = torch.transpose(x, 1, 2)
        x = self.fc1(x)
        x = torch.transpose(x, 1, 2)
        
        x1 = self.avgpool(x)
        x2 = self.globamaxpool(x)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x2 = torch.cat((x1, x2), 1)
#        print(x2.size(1))
        x = self.fc(x2)

        return x, x2


def myecgnet1(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

#def resnet34(pretrained=False, **kwargs):
#    """Constructs a ResNet-34 model.
#
#    Args:
#        pretrained (bool): If True, returns a model pre-trained on ImageNet
#    """
#    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#    return model

#if __name__ == '__main__':
##    import torch
#
#    x = torch.randn(1, 12, 5000)
#    m = myecgnet1()
#    m(x)
#    #    from torchvision.models import resnet
#    from torchsummary import summary
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
#    model = m.to(device)
#    summary(model, (12, 5000))