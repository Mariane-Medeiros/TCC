import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt


def calcular_media_desvio(data_loader):
    soma, soma_quadrado, n = 0, 0, 0
    for imgs, _ in data_loader:
        soma += torch.mean(imgs, dim=[0, 2, 3])
        soma_quadrado += torch.mean(imgs ** 2, dim=[0, 2, 3])
        n += 1
    media = soma / n
    desvio = (soma_quadrado / n - media ** 2).sqrt()
    return media, desvio


def preparar_dados(caminho_imagens, tamanho_imagem=(224, 224), batch_size=16, tamanho_validacao=0.2):
    # Transformação inicial para normalização
    transform_basico = transforms.Compose([
        transforms.Resize(tamanho_imagem),
        transforms.ToTensor(),
    ])

    dataset_temp = datasets.ImageFolder(
        root=caminho_imagens, transform=transform_basico)
    qtd_amostras = len(dataset_temp)

    # Dividir índices
    indices = list(range(qtd_amostras))
    np.random.seed(42)
    np.random.shuffle(indices)

    qtd_validacao = int(tamanho_validacao * qtd_amostras)
    indices_validacao = indices[:qtd_validacao]
    indices_treinamento = indices[qtd_validacao:]

    sample_treino = SubsetRandomSampler(indices_treinamento)
    sample_validacao = SubsetRandomSampler(indices_validacao)

    loader_temp = DataLoader(
        dataset_temp, batch_size=batch_size, sampler=sample_treino)

    # Calcular normalização
    media, desvio = calcular_media_desvio(loader_temp)

    # Transformação final com normalização
    transformacoes = transforms.Compose([
        transforms.Resize(tamanho_imagem),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=media, std=desvio)
    ])

    dataset = datasets.ImageFolder(
        root=caminho_imagens, transform=transformacoes)
    classes = dataset.classes

    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sample_treino)
    val_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sample_validacao)

    return train_loader, val_loader, classes, media, desvio
