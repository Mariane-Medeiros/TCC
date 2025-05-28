# IMPORTS
import numpy as np
import torch
import psutil
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# cuidar a quantidade de memoria que esta sendo usada, cuidado ao passar de 12:
# mem = psutil.virtual_memory()
# print(f"Memória usada: {mem.used / (1024 ** 3):.2f} GB")
# print(f"Disponível: {mem.available / (1024 ** 3):.2f} GB")

# CAMINHO DA IMAGEM
caminho_imagens = 'C:\Users\User\Downloads\tomate'

img_width = 256
img_height = 256


# TRANSOFMRAÇÕES DE PRE-PROCESSAMENTO
transformacoes = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# CARREGA O DATASET DO CAMINHO DEFINIDO E APLICA AS TRANSFORMACOES
dataset = datasets.ImageFolder(root=caminho_imagens, transform=transformacoes)

# VARIAVEIS PARA QUANTIDADE DE TESTE E VALIDACAO
tamanho_validacao = 0.1
qtd_amostras = len(dataset)
qtd_imagens_teste = tamanho_validacao * qtd_amostras
qtd_imagens_validacao = qtd_amostras - qtd_imagens_teste

# ARRAY COM TODOS OS MEUS INDICES EMBARALHADOS (VI QUE É BOM TER SEED PRA PODER SER REPRODUZIDO DEPOIS)
indices = list(range(qtd_amostras))
np.random.seed(42)
np.random.shuffle(indices)


# FAZER O SAMPLE PRA USAR NO DATALOADER
indices_teste = indices[:qtd_imagens_teste]
indices_validacao = indices[qtd_imagens_teste:]
sample_teste = SubsetRandomSampler(indices_teste)
sample_validacao = SubsetRandomSampler(indices_validacao)

# CARREGAR 2 DATALOADER, SubsetRandomSampler PRA QUE OS DADOS SEJAM MISTURADOS A CADA EPOCA
batch_tamanho = 16
teste_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_tamanho, sampler=sample_teste)
validacao_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_tamanho, sampler=sample_validacao)

# FAZER UM GET PRA PEGAR ESSES DATALOADER OU BOTAR TUDO DENTRO DE UMA FUNÇÃO E FAZER UM RETURN


#FUNÇÃO DE NORMALIZAÇÃO
def calcular_media_desvio(dataLoader):
    soma_canais, soma_canais_ao_quadrado, numero_batches = 0,0,0
    for imgs, _ in dataLoader:
        soma_canais += torch.mean(imgs, dim=[0,2,3]) #dim de dimensao, 0 é os batchs entao eu somo todas as imgs do batch, 2 e 3 "são altura e largura" 
        #soma_canais é um vetor com 3 valores, sendo a media de cada canal  
        soma_canais_ao_quadrado += torch.mean(imgs**2, dim=[0,2,3]) #serve pra calcular a variacia->desvio,esses dois são uma soma por causa do +=
        numero_batches += 1 #média das médias por batch. Isso é uma boa aproximação da média global
    media = soma_canais / numero_batches
    desvio = (soma_canais_ao_quadrado/numero_batches - media**2)*0.5#0.5 é a raiz e esse entre parenteses é a formula da variancia

    return media, desvio


media, desvio = calcular_media_desvio()
transforms.Normalize(mean=media, std=desvio)
