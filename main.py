import torch
import torch.nn as nn
import torch.optim as optim
from pre_processamento import preparar_dados
from avaliacao import avaliar_modelo
from modelos import VGG16, AlexNet


# ====== Configurações ======
caminho_imagens = 'C:/Users/User/Downloads/tcc/plantas'
tamanho_imagem = (224, 224)
batch_size = 16
tamanho_validacao = 0.2
numero_epocas = 10
taxa_aprendizado = 0.01
nome_arquivo_saida = "avaliacao_vgg16.txt"

# ====== Verificar dispositivo ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Usando dispositivo: {device}")

# ====== Pré-processamento ======
train_loader, val_loader, classes, media, desvio = preparar_dados(
    caminho_imagens=caminho_imagens,
    tamanho_imagem=tamanho_imagem,
    batch_size=batch_size,
    tamanho_validacao=tamanho_validacao
)

print(f"📊 Número de classes: {len(classes)} => {classes}")


# ====== Modelo ======
modelo = VGG16(num_classes=len(classes)).to(device)
criterio = nn.CrossEntropyLoss()
otimizador = optim.SGD(modelo.parameters(),
                       lr=taxa_aprendizado, momentum=0.9, weight_decay=0.001)

# ====== Treinamento ======
for epoca in range(numero_epocas):
    modelo.train()
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

    perda_media = perda_total / total_amostras
    acuracia = acertos / total_amostras * 100

    print(f"\n📚 Época {epoca + 1}/{numero_epocas}")
    print(
        f"🔧 Treino — Perda média: {perda_media:.4f} | Acurácia: {acuracia:.2f}%")

# ====== Avaliação ======
avaliar_modelo(modelo, val_loader, device, classes, nome_arquivo_saida)
