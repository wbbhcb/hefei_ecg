import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np


class Resnet_block(nn.Module):
    def __init__(self, inplanes, planes, planes2, kernel_size, stride=1, downsample=None):
        super(Resnet_block, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride, padding=0)
        self.leakyrelu = nn.LeakyReLU(0.3, inplace=True)
        # nn.Ze
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=stride, padding=0)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv1d(planes, planes2, kernel_size=1, stride=stride, padding=0)
        self.conv4 = nn.Conv1d(planes2, planes2, kernel_size=kernel_size, stride=stride, padding=0)
        self.conv5 = nn.Conv1d(planes2, planes2, kernel_size=1, stride=stride, padding=0)
        self.kernel_size = kernel_size

    def mypad(self, x, kernal_size, stride):
        out_size = np.ceil(x.size(2) / stride)
        padded_num = (out_size-1)*stride+kernal_size-x.size(2)
        if padded_num == 0:
            return x
        elif padded_num == 1:
            padded_data2 = torch.zeros((x.size(0), x.size(1), int(padded_num-int(padded_num/2))))
            if torch.cuda.is_available():
                padded_data2 = padded_data2.cuda()
            x = torch.cat((x, padded_data2), 2)
            return x
        else:
            padded_data1 = torch.zeros((x.size(0), x.size(1), int(padded_num/2)))
            padded_data2 = torch.zeros((x.size(0), x.size(1), int(padded_num-int(padded_num/2))))
            if torch.cuda.is_available():
                padded_data1 = padded_data1.cuda()
                padded_data2 = padded_data2.cuda()
            x = torch.cat((padded_data1, x, padded_data2), 2)
            return x

    def forward(self, x):
        out = self.conv1(x)
        out = self.leakyrelu(out)
        out = self.mypad(out, 2, 2)
        residual = self.maxpool(out)

        # print(residual.size)
        out = self.mypad(out, self.kernel_size, 1)
        out = self.conv2(out)
        out = self.leakyrelu(out)
        out = self.mypad(out, self.kernel_size, 1)
        out = self.conv2(out)
        out = self.leakyrelu(out)
        out = self.mypad(out, 2, 2)
        out = self.maxpool(out)
        out = out + residual

        for i in range(5):
            if i == 0:
                out = self.conv3(out)
            else:
                out = self.conv5(out)
            residual = self.mypad(out, 2, 2)
            residual = self.maxpool(residual)
            out = self.mypad(out, self.kernel_size, 1)
            out = self.conv4(out)
            out = self.leakyrelu(out)
            out = self.mypad(out, self.kernel_size, 1)
            out = self.conv4(out)
            out = self.leakyrelu(out)
            out = self.mypad(out, 2, 2)
            out = self.maxpool(out)
            out = out + residual
        return out


class CNN(nn.Module):
    def __init__(self, block, inplanes, num_classes=34):
        super(CNN, self).__init__()
        self.globavepool = nn.AdaptiveAvgPool1d(1)
        # self.block1 = block(inplanes, 16, 64, 2)
        self.block2 = block(inplanes, 16, 64, 4)
        # self.block3 = block(inplanes, 16, 64, 8)
        # self.block4 = block(inplanes, 16, 64, 16)
        # self.block5 = block(inplanes, 16, 64, 32)
        self.fc1 = nn.Linear(64, 256)
        self.fc_atten = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.3, inplace=True)
        self.fc2 = nn.Linear(256, num_classes)
        # self.sigmoid = nn.Sigmoid()

    def attention(self, x):
        out = torch.transpose(x, 1, 2)
        out = self.fc_atten(out)
        out = torch.transpose(out, 1, 2)
        weight = self.globavepool(out)
        out = weight * x
        return out

    def forward(self, x):
        # out1 = self.block1(x)
        # out1 = self.attention(out1)
        out2 = self.block2(x)
        # out2 = self.attention(out2)
        # out3 = self.block3(x)
        # out3 = self.attention(out3)
        # out4 = self.block4(x)
        # out4 = self.attention(out4)
        # out5 = self.block5(x)
        # out5 = self.attention(out5)
        # out = torch.cat((out1, out2, out3, out4, out5), 1)
        out = out2
        time_step = out.size(2)
        print(out.size(1))
        out = torch.transpose(out, 1, 2)
        # out = torch.reshape(out, (-1, out.size(2)))
        out = self.fc1(out)
        # out = torch.reshape(out, (-1, time_step, out.size(1)))
        out = torch.transpose(out, 1, 2)
        out2 = self.globavepool(out)

        out = torch.reshape(out2, (-1, out2.size(1)))
        out = self.dropout(out)
        out = self.fc2(out)
        # out = self.sigmoid(out)
        # out.backward()
        return out # , out2





def myCNN(**kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CNN(Resnet_block, 8, **kwargs)
    return model

#from torchsummary import summary
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
#model = CNN(Resnet_block, 8).to(device)
#summary(model, (8, 2049))
