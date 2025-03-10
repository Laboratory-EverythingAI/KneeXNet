#!/usr/bin/env python3.6

import torch
import torch.nn as nn
from torchvision import models

#VGG16
class VGGNet(nn.Module):
    def __init__(self):
        super().__init__()
        ################################
        #在这第一个是resnet18，第二个是convnet_small,
        ##############################
        self.resnet = models.vgg16(pretrained=True)
        # 修改分类器的输入大小以适应 ResNet 的输出大小
        self.fc = nn.Linear(512, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 使用自适应平均池化层
        self.dropout = nn.Dropout(p=0.5)

    @property
    def features(self):
        # 返回 ResNet 的特征提取部分
        return nn.Sequential(*list(self.resnet.children())[:-1])

    @property
    def classifier(self):
        # 返回分类器
        return self.fc

    def forward(self, batch):
        batch_out = torch.tensor([]).to(batch.device)

        for series in batch:
            out = torch.tensor([]).to(batch.device)
            for image in series:
                out = torch.cat((out, self.features(image.unsqueeze(0))), 0)

            out = self.avg_pool(out).squeeze()
            out = out.max(dim=0, keepdim=True)[0].squeeze()

            out = self.classifier(self.dropout(out))

            batch_out = torch.cat((batch_out, out), 0)

        return batch_out

#ResNET
class ResNet18Net(nn.Module):
    def __init__(self):
        super().__init__()
        ################################
        #在这第一个是resnet18，第二个是convnet_small,
        ##############################
        self.resnet = models.resnet18(pretrained=True)
        # 修改分类器的输入大小以适应 ResNet 的输出大小
        self.fc = nn.Linear(512, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 使用自适应平均池化层
        self.dropout = nn.Dropout(p=0.5)

    @property
    def features(self):
        # 返回 ResNet 的特征提取部分
        return nn.Sequential(*list(self.resnet.children())[:-1])

    @property
    def classifier(self):
        # 返回分类器
        return self.fc

    def forward(self, batch):
        batch_out = torch.tensor([]).to(batch.device)

        for series in batch:
            out = torch.tensor([]).to(batch.device)
            for image in series:
                out = torch.cat((out, self.features(image.unsqueeze(0))), 0)

            out = self.avg_pool(out).squeeze()
            out = out.max(dim=0, keepdim=True)[0].squeeze()

            out = self.classifier(self.dropout(out))

            batch_out = torch.cat((batch_out, out), 0)

        return batch_out

#####################
class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        #################################
        #第一个是：alexnet；第二个是resnet18,；第三个是：
        #################################
        self.alexnet = models.alexnet(pretrained=True).features
        self.fc = nn.Linear(256, 1)

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=None, padding=0)
        self.dropout = nn.Dropout(p=0.5)

    @property
    def features(self):
        return self.alexnet

    @property
    def classifier(self):
        return self.fc

    def forward(self, batch):
        batch_out = torch.tensor([]).to(batch.device)

        for series in batch:
            out = torch.tensor([]).to(batch.device)
            for image in series:
                out = torch.cat((out, self.features(image.unsqueeze(0))), 0)

            out = self.avg_pool(out).squeeze()
            out = out.max(dim=0, keepdim=True)[0].squeeze()

            out = self.classifier(self.dropout(out))

            batch_out = torch.cat((batch_out, out), 0)

        return batch_out

#convnext
class convextNet(nn.Module):
    def __init__(self):
        super().__init__()
        #################################
        #第一个是：alexnet；第二个是resnet18,；第三个是：
        #################################
        self.alexnet = models.convnext_tiny(pretrained=True).features
        self.fc = nn.Linear(768, 1)

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=None, padding=0)
        self.dropout = nn.Dropout(p=0.5)

    @property
    def features(self):
        return self.alexnet

    @property
    def classifier(self):
        return self.fc

    def forward(self, batch):
        batch_out = torch.tensor([]).to(batch.device)

        for series in batch:
            out = torch.tensor([]).to(batch.device)
            for image in series:
                out = torch.cat((out, self.features(image.unsqueeze(0))), 0)

            out = self.avg_pool(out).squeeze()
            out = out.max(dim=0, keepdim=True)[0].squeeze()

            out = self.classifier(self.dropout(out))

            batch_out = torch.cat((batch_out, out), 0)

        return batch_out