import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from pre_processamento import preparar_dados
from avaliacao import avaliar_modelo
from modelos import VGG16, AlexNet, MobileNet, ResNet
from modelos_transfer_learning import tl_VGG16, tl_AlexNet, tl_MobileNet, tl_ResNet

# ====== Configurações ======

caminho_imagens = 'C:/Users/User/Downloads/tcc/desenvolvimento/plantas'
tamanho_imagem = (224, 224)
batch_size = 16
tamanho_validacao = 0.2
numero_epocas = 20
taxa_aprendizado = 0.0001

# ====== Verificar dispositivo ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Inicialização de pesos personalizada ======


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# ====== Pré-processamento ======
train_loader, val_loader, classes, media, desvio = preparar_dados(
    caminho_imagens=caminho_imagens,
    tamanho_imagem=tamanho_imagem,
    batch_size=batch_size,
    tamanho_validacao=tamanho_validacao
)

# Verificar e imprimir a ordem das classes
print("Ordem das classes:", classes)

lista_de_rotulos = []
for _, rotulos in train_loader:
    lista_de_rotulos.extend(rotulos.numpy())

# ====== Calcular pesos balanceados ======
pesos = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(classes)),
    y=lista_de_rotulos
)
pesos_tensor = torch.FloatTensor(pesos).to(device)

# ====== Modelo ======
modelo = tl_AlexNet(num_classes=len(classes)).to(device)

modelo.apply(init_weights)

criterio = nn.CrossEntropyLoss(weight=pesos_tensor)
otimizador = optim.SGD(modelo.parameters(),
                       lr=taxa_aprendizado, momentum=0.9, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(otimizador, step_size=5, gamma=0.5)

# ====== Obter nome de arquivo de saída do modelo ======
nome_arquivo_saida = getattr(
    modelo, 'nome_arquivo_saida', 'avaliacao_modelo') + ".txt"
nome_log_treinamento = nome_arquivo_saida.replace(".txt", "_treinamento.txt")

# ====== Início do tempo total ======
inicio_tempo_total = time.time()

# ====== Treinamento ======
with open(nome_log_treinamento, "w", encoding="utf-8") as log_file:
    for epoca in range(numero_epocas):
        log_file.write(f"\n Época {epoca + 1}/{numero_epocas}\n")
        print(f"\n Época {epoca + 1}/{numero_epocas}")

        modelo.train()
        inicio_treino = time.time()

        perda_total = 0
        acertos = 0
        total_amostras = 0

        for imagens, rotulos in train_loader:
            imagens, rotulos = imagens.to(device), rotulos.to(device)
            otimizador.zero_grad()
            saidas = modelo(imagens)
            perda = criterio(saidas, rotulos)
            perda.backward()
            otimizador.step()

            perda_total += perda.item() * imagens.size(0)
            _, previsoes = torch.max(saidas, 1)
            acertos += (previsoes == rotulos).sum().item()
            total_amostras += rotulos.size(0)

        fim_treino = time.time()
        tempo_treino = fim_treino - inicio_treino
        perda_media = perda_total / total_amostras
        acuracia = acertos / total_amostras * 100

        print(
            f" Treino — Perda: {perda_media:.4f} | Acurácia: {acuracia:.2f}% | ⏱ Tempo: {tempo_treino:.2f}s")
        log_file.write(
            f" Treino — Perda: {perda_media:.4f} | Acurácia: {acuracia:.2f}% | ⏱ Tempo: {tempo_treino:.2f}s\n")

        # ====== Validação ======
        modelo.eval()
        inicio_val = time.time()

        perda_total_val = 0
        acertos_val = 0
        total_val = 0

        with torch.no_grad():
            for imagens, rotulos in val_loader:
                imagens, rotulos = imagens.to(device), rotulos.to(device)
                saidas = modelo(imagens)
                perda = criterio(saidas, rotulos)

                perda_total_val += perda.item() * imagens.size(0)
                _, previsoes = torch.max(saidas, 1)
                acertos_val += (previsoes == rotulos).sum().item()
                total_val += rotulos.size(0)

        fim_val = time.time()
        tempo_val = fim_val - inicio_val
        perda_val = perda_total_val / total_val
        acuracia_val = acertos_val / total_val * 100

        print(
            f" Validação — Perda: {perda_val:.4f} | Acurácia: {acuracia_val:.2f}% | ⏱ Tempo: {tempo_val:.2f}s")
        log_file.write(
            f" Validação — Perda: {perda_val:.4f} | Acurácia: {acuracia_val:.2f}% | ⏱ Tempo: {tempo_val:.2f}s\n")

        scheduler.step()

# ====== Tempo total ======
fim_tempo_total = time.time()
tempo_total = fim_tempo_total - inicio_tempo_total
print(f"\n Tempo total de treinamento: {tempo_total:.2f} segundos")
with open(nome_log_treinamento, "a", encoding="utf-8") as log_file:
    log_file.write(
        f"\n Tempo total de treinamento: {tempo_total:.2f} segundos\n")

# ====== Avaliação final ======
avaliar_modelo(modelo, val_loader, device, classes, nome_arquivo_saida)
