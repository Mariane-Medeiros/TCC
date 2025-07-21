import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import gc

caminho_imagens = 'C:/Users/User/Downloads/tcc/plantas'

img_width = 256
img_height = 256
batch_tamanho = 16
tamanho_validacao = 0.2
numero_classes = 6
numero_epocas = 20
batch_size = 16
taxa_aprendizado = 0.01

transform_basico = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset_temp = datasets.ImageFolder(
    root=caminho_imagens, transform=transform_basico)

qtd_amostras = len(dataset_temp)
qtd_imagens_teste = int(tamanho_validacao * qtd_amostras)
qtd_imagens_validacao = qtd_amostras - qtd_imagens_teste

indices = list(range(qtd_amostras))
np.random.seed(42)
np.random.shuffle(indices)

indices_teste = indices[:qtd_imagens_teste]
indices_validacao = indices[qtd_imagens_teste:]

sample_teste = SubsetRandomSampler(indices_teste)
sample_validacao = SubsetRandomSampler(indices_validacao)

teste_loader_temp = DataLoader(
    dataset_temp, batch_size=batch_tamanho, sampler=sample_teste)


def calcular_media_desvio(dataLoader):
    soma_canais, soma_canais_ao_quadrado, numero_batches = 0, 0, 0
    for imgs, _ in dataLoader:
        soma_canais += torch.mean(imgs, dim=[0, 2, 3])
        soma_canais_ao_quadrado += torch.mean(imgs**2, dim=[0, 2, 3])
        numero_batches += 1
    media = soma_canais / numero_batches
    desvio = (soma_canais_ao_quadrado / numero_batches - media**2).sqrt()
    return media, desvio


media, desvio = calcular_media_desvio(teste_loader_temp)


transformacoes = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=media, std=desvio),
])

dataset = datasets.ImageFolder(root=caminho_imagens, transform=transformacoes)


treinamento_loader = DataLoader(
    dataset, batch_size=batch_tamanho, sampler=sample_teste)
validacao_loader = DataLoader(
    dataset, batch_size=batch_tamanho, sampler=sample_validacao)


def mostrar_batch(data_loader, dataset):
    for imagens, rotulos in data_loader:
        print(f'Formato das imagens no batch: {imagens.shape}')
        print(f'Rótulos do batch: {rotulos.tolist()}')

        img = imagens[0].permute(1, 2, 0)
        img = img * desvio.detach().clone() + media.detach().clone()
        img = img.clamp(0, 1)

        plt.imshow(img)
        nome_classe = dataset.classes[rotulos[0]]
        plt.title(f'Classe: {nome_classe}')
        plt.axis('off')
        plt.show()
        break


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
        input = x
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.conv2(output)
        output = self.bn2(output)
        if self.downsample:
            input = self.downsample(x)
        output += input
        output = self.relu(output)
        return output


class ResNet(nn.Module):
    def __init__(self, bloco, camadas, numero_classes):
        super().__init__()

        self.canais_entrada = 64

        self.conv1 = nn.Conv2d(3, self.canais_entrada, kernel_size=7,
                               stride=2, padding=3, bias=False)  # 3 canais inicias -> rgb
        self.bn1 = nn.BatchNorm2d(num_features=self.canais_entrada)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.camada0 = self.criar_camada(bloco, 64, camadas[0], stride=1)
        self.camada1 = self.criar_camada(bloco, 128, camadas[1], stride=2)
        self.camada2 = self.criar_camada(bloco, 256, camadas[2], stride=2)
        self.camada3 = self.criar_camada(bloco, 512, camadas[3], stride=2)

        self.adppool = nn.AdaptiveAvgPool2d((1, 1))
        self.classificacao = nn.Linear(
            in_features=512, out_features=numero_classes)

    def criar_camada(self, bloco, canais_saida, blocos, stride=1):
        downsample = None

        if stride != 1 or self.canais_entrada != canais_saida:
            downsample = nn.Sequential(
                nn.Conv2d(self.canais_entrada, canais_saida,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(canais_saida),
            )

        camadas = []
        camadas.append(bloco(self.canais_entrada,
                       canais_saida, stride, downsample))
        self.canais_entrada = canais_saida

        for i in range(1, blocos):
            camadas.append(bloco(self.canais_entrada, canais_saida))

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

        x = self.adppool(x)
        x = torch.flatten(x, 1)
        x = self.classificacao(x)

        return x


device = torch.device("cpu")
model = ResNet(BlocoResidual, [2, 2, 2, 2], 6).to(device)
funcao_perda = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=taxa_aprendizado, weight_decay=0.001, momentum=0.9)

total_steps = len(validacao_loader)


for epoca in range(numero_epocas):
    print(f"\n Época {epoca + 1}/{numero_epocas}")
    model.train()

    total_loss_treino = 0
    total_corretos_treino = 0
    total_amostras_treino = 0

    for imagens, rotulos in treinamento_loader:
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

        del imagens, rotulos, saidas, perda, previsoes
        torch.cuda.empty_cache()
        gc.collect()

    perda_media_treino = total_loss_treino / total_amostras_treino
    acuracia_treino = total_corretos_treino / total_amostras_treino * 100
    print(
        f" Treinamento — Perda média: {perda_media_treino:.4f} | Acurácia: {acuracia_treino:.2f}%")

    total_loss_validacao = 0
    total_corretos_validacao = 0
    total_amostras_validacao = 0

    with torch.no_grad():
        for imagens, rotulos in validacao_loader:
            imagens, rotulos = imagens.to(device), rotulos.to(device)

            saidas = model(imagens)
            perda = funcao_perda(saidas, rotulos)
            total_loss_validacao += perda.item() * imagens.size(0)

            _, previsoes = torch.max(saidas, 1)
            total_corretos_validacao += (previsoes == rotulos).sum().item()
            total_amostras_validacao += rotulos.size(0)
            del imagens, rotulos, saidas, perda, previsoes
            torch.cuda.empty_cache()
            gc.collect()

    perda_media_validacao = total_loss_validacao / total_amostras_validacao
    acuracia_validacao = total_corretos_validacao / total_amostras_validacao * 100
    print(
        f" Validação — Perda média: {perda_media_validacao:.4f} | Acurácia: {acuracia_validacao:.2f}%")
    torch.save(model.state_dict(), 'modelo_epoca_resnet3.pth')

torch.save(model.state_dict(), 'modelo_final3.pth')
