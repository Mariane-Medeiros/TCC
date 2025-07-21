import torch
import torch.nn as nn
import torch.nn.functional as F


class Steam(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.branch1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=0),
        self.branch2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        self.branch3 = nn.Conv2d(64, 80, kernel_size=1, stride=1, padding=0),
        self.branch4 = nn.Conv2d(80, 192, kernel_size=3, stride=1, padding=1),
        self.branch5 = nn.MaxPool2d(kernel_size=3, stride=2),


class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 64, kernel_size=1)

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=1),
            nn.Conv2d(48, 64, kernel_size=5, padding=2)
        )

        self.branch3x3dbl = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.Conv2d(96, 96, kernel_size=3, padding=1)
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 32, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1x1(x),
            self.branch5x5(x),
            self.branch3x3dbl(x),
            self.branch_pool(x)
        ], 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 192, kernel_size=1)

        self.branch7x7 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.Conv2d(128, 128, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(128, 192, kernel_size=(7, 1), padding=(3, 0))
        )

        self.branch7x7dbl = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.Conv2d(128, 128, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(128, 128, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(128, 128, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(128, 192, kernel_size=(1, 7), padding=(0, 3))
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 192, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1x1(x),
            self.branch7x7(x),
            self.branch7x7dbl(x),
            self.branch_pool(x)
        ], 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 320, kernel_size=1)

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 384, kernel_size=1),
        )
        self.branch3x3_1 = nn.Conv2d(
            384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2 = nn.Conv2d(
            384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl = nn.Sequential(
            nn.Conv2d(in_channels, 448, kernel_size=1),
            nn.Conv2d(448, 384, kernel_size=3, padding=1)
        )
        self.branch3x3dbl_1 = nn.Conv2d(
            384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_2 = nn.Conv2d(
            384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 192, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1x1(x)

        b2 = self.branch3x3(x)
        b2 = torch.cat([self.branch3x3_1(b2), self.branch3x3_2(b2)], 1)

        b3 = self.branch3x3dbl(x)
        b3 = torch.cat([self.branch3x3dbl_1(b3), self.branch3x3dbl_2(b3)], 1)

        b4 = self.branch_pool(x)

        return torch.cat([b1, b2, b3, b4], 1)


class ReductionA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch3x3 = nn.Conv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.Conv2d(96, 96, kernel_size=3, stride=2)
        )

        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        return torch.cat([
            self.branch3x3(x),
            self.branch3x3dbl(x),
            self.branch_pool(x)
        ], 1)


class ReductionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=1),
            nn.Conv2d(192, 320, kernel_size=3, stride=2)
        )

        self.branch7x7x3 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=1),
            nn.Conv2d(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(192, 192, kernel_size=3, stride=2)
        )

        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        return torch.cat([
            self.branch1x1(x),
            self.branch7x7x3(x),
            self.branch_pool(x)
        ], 1)


class Inceptionv3(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),  # 149x149
            nn.Conv2d(32, 32, kernel_size=3),           # 147x147
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2),                  # 73x73
            nn.Conv2d(64, 80, kernel_size=1),
            nn.Conv2d(80, 192, kernel_size=3),
            nn.MaxPool2d(3, stride=2)                   # 35x35
        )

        self.inception_a = nn.Sequential(
            InceptionA(192),
            InceptionA(256),
            InceptionA(288)
        )

        self.reduction_a = ReductionA(288)

        self.inception_b = nn.Sequential(
            InceptionB(768),
            InceptionB(768),
            InceptionB(768),
            InceptionB(768)
        )

        self.reduction_b = ReductionB(768)

        self.inception_c = nn.Sequential(
            InceptionC(1280),
            InceptionC(2048),
            InceptionC(2048)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
