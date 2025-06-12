import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import gc
import numpy as np
import os

# Caminho onde as imagens estão e o modelo foi salvo
caminho_imagens = 'C:/Users/User/Downloads/tomate'
caminho_modelo = 'modelo_final2.pth'  # ou 'modelo_epoca_resnet2.pth'

# Parâmetros
batch_tamanho = 16
device = torch.device("cpu")  # ou "cuda" se estiver usando GPU

# Transformações (usando média e desvio calculados previamente)
# Recalcular média e desvio como antes:


def calcular_acuracia(modelo, dataloader, device):
    modelo.eval()
    total = 0
    corretos = 0

    with torch.no_grad():
        for imagens, rotulos in dataloader:
            imagens, rotulos = imagens.to(device), rotulos.to(device)
            saidas = modelo(imagens)
            _, previsoes = torch.max(saidas, 1)
            corretos += (previsoes == rotulos).sum().item()
            total += rotulos.size(0)

    acuracia = 100 * corretos / total
    print(
        f"\nAcurácia do modelo no conjunto de dados: {acuracia:.2f}% ({corretos}/{total} acertos)")


def calcular_media_desvio(dataLoader):
    soma_canais, soma_canais_ao_quadrado, numero_batches = 0, 0, 0
    for imgs, _ in dataLoader:
        soma_canais += torch.mean(imgs, dim=[0, 2, 3])
        soma_canais_ao_quadrado += torch.mean(imgs**2, dim=[0, 2, 3])
        numero_batches += 1
    media = soma_canais / numero_batches
    desvio = (soma_canais_ao_quadrado / numero_batches - media**2).sqrt()
    return media, desvio


# Primeiro dataset temporário para calcular média/desvio
transform_basico = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
dataset_temp = datasets.ImageFolder(
    root=caminho_imagens, transform=transform_basico)
dataloader_temp = DataLoader(dataset_temp, batch_size=batch_tamanho)
media, desvio = calcular_media_desvio(dataloader_temp)

# Dataset final com normalização
transformacoes = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=media, std=desvio),
])
dataset = datasets.ImageFolder(root=caminho_imagens, transform=transformacoes)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)


# Define as classes do modelo


class BlocoResidual(torch.nn.Module):
    def __init__(self, canais_entrada, canais_saida, stride=1, downsample=None):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            canais_entrada, canais_saida, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(canais_saida)
        self.conv2 = torch.nn.Conv2d(
            canais_saida, canais_saida, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(canais_saida)
        self.relu = torch.nn.ReLU()
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


class ResNet(torch.nn.Module):
    def __init__(self, bloco, camadas, numero_classes):
        super().__init__()
        self.canais_entrada = 64
        self.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.camada0 = self.criar_camada(bloco, 64, camadas[0])
        self.camada1 = self.criar_camada(bloco, 128, camadas[1], stride=2)
        self.camada2 = self.criar_camada(bloco, 256, camadas[2], stride=2)
        self.camada3 = self.criar_camada(bloco, 512, camadas[3], stride=2)
        self.adppool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classificacao = torch.nn.Linear(512, 6)

    def criar_camada(self, bloco, canais_saida, blocos, stride=1):
        downsample = None
        if stride != 1 or self.canais_entrada != canais_saida:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.canais_entrada, canais_saida,
                                kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(canais_saida),
            )
        camadas = [
            bloco(self.canais_entrada, canais_saida, stride, downsample)]
        self.canais_entrada = canais_saida
        for _ in range(1, blocos):
            camadas.append(bloco(canais_saida, canais_saida))
        return torch.nn.Sequential(*camadas)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.camada0(x)
        x = self.camada1(x)
        x = self.camada2(x)
        x = self.camada3(x)
        x = self.adppool(x)
        x = torch.flatten(x, 1)
        x = self.classificacao(x)
        return x


# Carregar o modelo
modelo = ResNet(BlocoResidual, [2, 2, 2, 2], 6).to(device)
modelo.load_state_dict(torch.load(caminho_modelo, map_location=device))
modelo.eval()
calcular_acuracia(modelo, dataloader, device)

# Chamar a função após carregar o modelo

# Testar o modelo
with torch.no_grad():
    for imagens, rotulos in dataloader:
        imagens = imagens.to(device)
        saidas = modelo(imagens)
        _, previsao = torch.max(saidas, 1)

        # Reverter normalização para exibir imagem
        img = imagens[0].cpu().clone()
        img = img * desvio[:, None, None] + media[:, None, None]
        img = img.permute(1, 2, 0).clamp(0, 1)

        plt.imshow(img)
        plt.title(
            f"Verdadeira: {dataset.classes[rotulos[0]]} | Prevista: {dataset.classes[previsao[0]]}")
        plt.axis('off')
        plt.show()

        break  # Remova este break se quiser testar mais de uma imagem
