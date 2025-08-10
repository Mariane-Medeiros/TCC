import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class tl_VGG16(nn.Module):
    nome_arquivo_saida = "tl_validacao_vgg16"

    def __init__(self, num_classes):
        super(tl_VGG16, self).__init__()
        modelo_pre_treinado = models.vgg16(pretrained=True)
        for param in modelo_pre_treinado.features.parameters():
            param.requires_grad = False  # Congela as features

        self.features = modelo_pre_treinado.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # mesma do VGG
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class tl_AlexNet(nn.Module):
    nome_arquivo_saida = "tl_validacao_alexnet"

    def __init__(self, num_classes):
        super(tl_AlexNet, self).__init__()
        modelo_pre_treinado = models.alexnet(pretrained=True)

        # Congelar todas as camadas inicialmente
        for param in modelo_pre_treinado.features.parameters():
            param.requires_grad = False

        # Descongelar as últimas camadas convolucionais
        # Descongelar as últimas 8 camadas, por exemplo
        for i in range(-8, 0):
            list(modelo_pre_treinado.features.parameters())[
                i].requires_grad = True

        self.features = modelo_pre_treinado.features

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class tl_MobileNet(nn.Module):
    nome_arquivo_saida = "tl_validacao_mobilenet"

    def __init__(self, num_classes):
        super(tl_MobileNet, self).__init__()
        modelo_pre_treinado = models.mobilenet_v2(pretrained=True)
        for param in modelo_pre_treinado.features.parameters():
            param.requires_grad = False

        self.features = modelo_pre_treinado.features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(modelo_pre_treinado.last_channel, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # global average pooling
        x = self.classifier(x)
        return x


class tl_ResNet(nn.Module):
    nome_arquivo_saida = "tl_validacao_resnet"

    def __init__(self, num_classes):
        super(tl_ResNet, self).__init__()
        modelo_pre_treinado = models.resnet18(pretrained=True)
        for param in modelo_pre_treinado.parameters():
            param.requires_grad = False

        self.features = nn.Sequential(
            *list(modelo_pre_treinado.children())[:-1])  # remove fc
        self.fc = nn.Linear(modelo_pre_treinado.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


__all__ = ['tl_VGG16', 'tl_AlexNet', 'tl_MobileNet', 'tl_ResNet']
