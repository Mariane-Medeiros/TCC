import numpy as np
import torch
import torch.nn as nn

from pre_processamento import preparar_dados
caminho_imagens = 'C:/Users/User/Downloads/tcc/plantas'

treino_loader, valid_loader, classes, media, desvio = preparar_dados(
    caminho_imagens=caminho_imagens,
    tamanho_imagem=(224, 224),
    batch_size=16,
    tamanho_validacao=0.2
)

num_classes = 4


class AlexNet(nn.Module):
    nome_arquivo_saida = "amora_validacao_alexnet"

    def __init__(self, num_classes=num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # bloco 1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # bloco 2
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # bloco 3
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            # bloco 4
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            # bloco 5
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            # bloco 6
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(True),
            # bloco 7
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            # bloco 8
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGG16(nn.Module):
    nome_arquivo_saida = "amora_validacao_vgg16"

    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            # bloco 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # bloco 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # bloco 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # bloco 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # bloco 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)           # Extração de features
        x = self.avgpool(x)            # Reduz para (512, 1, 1)
        x = torch.flatten(x, 1)        # Fica (batch_size, 512)
        x = self.classifier(x)         # Classificação final
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, kernel_size=3,
                                   stride=stride, padding=1,
                                   groups=in_c, bias=False)
        self.pointwise = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MobileNet(nn.Module):
    nome_arquivo_saida = "amora_validacao_mobilenet"

    def __init__(self, num_classes=4):
        super(MobileNet, self).__init__()

        self.features = nn.Sequential(
            # Conv / s2: 3x3x3x32
            nn.Conv2d(3, 32, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Conv dw / s1 + Conv 1x1
            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
            DepthwiseSeparableConv(256, 512, stride=2),

            # 5x: 512 → 512
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),

            DepthwiseSeparableConv(512, 1024, stride=2),
            DepthwiseSeparableConv(1024, 1024, stride=1),

            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


__all__ = ['ResNet']


class BlocoResidual(nn.Module):
    def __init__(self, canais_entrada, canais_saida, stride=1, downsample=None):
        super(BlocoResidual, self).__init__()
        self.conv1 = nn.Conv2d(canais_entrada, canais_saida,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(canais_saida)
        self.conv2 = nn.Conv2d(canais_saida, canais_saida,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(canais_saida)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    nome_arquivo_saida = "amora_validacao_resnet"

    def __init__(self, num_classes):
        super().__init__()
        bloco = BlocoResidual
        camadas = [2, 2, 2, 2]

        self.canais_entrada = 64

        self.conv1 = nn.Conv2d(3, self.canais_entrada, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.canais_entrada)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.camada0 = self._criar_camada(bloco, 64, camadas[0])
        self.camada1 = self._criar_camada(bloco, 128, camadas[1], stride=2)
        self.camada2 = self._criar_camada(bloco, 256, camadas[2], stride=2)
        self.camada3 = self._criar_camada(bloco, 512, camadas[3], stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _criar_camada(self, bloco, canais_saida, blocos, stride=1):
        downsample = None
        if stride != 1 or self.canais_entrada != canais_saida:
            downsample = nn.Sequential(
                nn.Conv2d(self.canais_entrada, canais_saida,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(canais_saida),
            )

        camadas = [
            bloco(self.canais_entrada, canais_saida, stride, downsample)]
        self.canais_entrada = canais_saida

        for _ in range(1, blocos):
            camadas.append(bloco(canais_saida, canais_saida))

        return nn.Sequential(*camadas)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.camada0(x)
        x = self.camada1(x)
        x = self.camada2(x)
        x = self.camada3(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


__all__ = ['VGG16', 'AlexNet', 'MobileNet', 'ResNet']
