import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
import gc
import os

# ====== CONFIGURAÇÃO ======
caminho_imagens = 'C:/Users/User/Downloads/TCC-main/TCC-main/soja'
batch_tamanho = 16
tamanho_validacao = 0.2
numero_epocas = 20
taxa_aprendizado = 0.01

# ====== AUMENTO DE DADOS ======
transformacoes = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

# ====== CARREGAMENTO TEMPORÁRIO PARA NORMALIZAÇÃO ======
dataset_temp = datasets.ImageFolder(
    root=caminho_imagens, transform=transformacoes)
treinamento_loader_temp = DataLoader(
    dataset_temp, batch_size=batch_tamanho, shuffle=False)

# ====== CÁLCULO DE MÉDIA E DESVIO ======


def calcular_media_desvio(dataLoader):
    soma, soma_quadrado, n = 0, 0, 0
    for imgs, _ in dataLoader:
        soma += torch.mean(imgs, dim=[0, 2, 3])
        soma_quadrado += torch.mean(imgs ** 2, dim=[0, 2, 3])
        n += 1
    media = soma / n
    desvio = (soma_quadrado / n - media ** 2).sqrt()
    return media, desvio


media, desvio = calcular_media_desvio(treinamento_loader_temp)

# ====== TRANSFORMAÇÃO FINAL COM NORMALIZAÇÃO ======
transformacoes = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=media, std=desvio),
])

dataset = datasets.ImageFolder(root=caminho_imagens, transform=transformacoes)
numero_classes = len(dataset.classes)

# ====== SPLIT DE DADOS ======
qtd_amostras = len(dataset)
indices = list(range(qtd_amostras))
np.random.seed(42)
np.random.shuffle(indices)

qtd_validacao = int(tamanho_validacao * qtd_amostras)
indices_validacao = indices[:qtd_validacao]
indices_treinamento = indices[qtd_validacao:]

sample_treino = SubsetRandomSampler(indices_treinamento)
sample_validacao = SubsetRandomSampler(indices_validacao)

train_loader = DataLoader(
    dataset, batch_size=batch_tamanho, sampler=sample_treino)
val_loader = DataLoader(
    dataset, batch_size=batch_tamanho, sampler=sample_validacao)

# ====== MODELO COM DROPOUT ======


class BlocoResidual(nn.Module):
    def __init__(self, canais_entrada, canais_saida, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(canais_entrada, canais_saida,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(canais_saida)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.3)
        self.conv2 = nn.Conv2d(canais_saida, canais_saida,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(canais_saida)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, bloco, camadas, numero_classes):
        super().__init__()
        self.canais_entrada = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.camada0 = self.criar_camada(bloco, 64, camadas[0])
        self.camada1 = self.criar_camada(bloco, 128, camadas[1], stride=2)
        self.camada2 = self.criar_camada(bloco, 256, camadas[2], stride=2)
        self.camada3 = self.criar_camada(bloco, 512, camadas[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, numero_classes)

    def criar_camada(self, bloco, canais_saida, blocos, stride=1):
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
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ====== TREINAMENTO ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet(BlocoResidual, [2, 2, 2, 2], numero_classes).to(device)
funcao_perda = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=taxa_aprendizado, weight_decay=0.001, momentum=0.9)

for epoca in range(numero_epocas):
    model.train()
    total_loss_treino = 0
    total_corretos_treino = 0
    total_amostras_treino = 0

    for imagens, rotulos in train_loader:
        imagens, rotulos = imagens.to(device), rotulos.to(device)
        optimizer.zero_grad()
        saidas = model(imagens)
        perda = funcao_perda(saidas, rotulos)
        perda.backward()
        optimizer.step()

        total_loss_treino += perda.item() * imagens.size(0)
        _, previsoes = torch.max(saidas, 1)
        total_corretos_treino += (previsoes == rotulos).sum().item()
        total_amostras_treino += rotulos.size(0)

    perda_media_treino = total_loss_treino / total_amostras_treino
    acuracia_treino = total_corretos_treino / total_amostras_treino * 100

    model.eval()
    total_loss_validacao = 0
    total_corretos_validacao = 0
    total_amostras_validacao = 0

    with torch.no_grad():
        for imagens, rotulos in val_loader:
            imagens, rotulos = imagens.to(device), rotulos.to(device)
            saidas = model(imagens)
            perda = funcao_perda(saidas, rotulos)
            total_loss_validacao += perda.item() * imagens.size(0)
            _, previsoes = torch.max(saidas, 1)
            total_corretos_validacao += (previsoes == rotulos).sum().item()
            total_amostras_validacao += rotulos.size(0)

    perda_media_validacao = total_loss_validacao / total_amostras_validacao
    acuracia_validacao = total_corretos_validacao / total_amostras_validacao * 100

    print(f"\n Época {epoca + 1}/{numero_epocas}")
    print(
        f" Treinamento — Perda média: {perda_media_treino:.4f} | Acurácia: {acuracia_treino:.2f}%")
    print(
        f" Validação — Perda média: {perda_media_validacao:.4f} | Acurácia: {acuracia_validacao:.2f}%")
