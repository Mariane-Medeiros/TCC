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

# FAZER A FUNÇÃO DE NORMALIZAÇÃO
img_width = 256
img_height = 256


# *******PROBLEMAS: NAO CONSIGO USAR MEAN E STD FORA DA FUNÇÃO, ESTOU CALCULANDO PRA UMA IMG ALEATORIA E PRECISO CALCULAR PRA TODO DATABASE**********
def calcular_media_desvio():
    img = torch.randn(3, img_height, img_width)
    media_r = img[0].mean().item()
    media_g = img[1].mean().item()
    media_b = img[2].mean().item()
    desvio_r = img[0].std().item()
    desvio_g = img[1].std().item()
    desvio_b = img[2].std().item()
    mean = [media_r, media_g, media_b]
    std = [desvio_r, desvio_g, desvio_b]


# TRANSOFMRAÇÕES DE PRE-PROCESSAMENTO
transformacoes = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# CARREGA O DATASET DO CAMINHO DEFINIDO E APLICA AS TRANSFORMACOES
dataset = datasets.ImageFolder(root=caminho_imagens, transform=transformacoes)

# VARIAVEIS PARA QUANTIDADE DE TESTE E VALIDACAO
tamanho_validacao = 0.1
qtd_amostras = len(dataset)
qtd_imagens_teste = tamanho_validacao * qtd_amostras
qtd_imagens_validacao = qtd_amostras - qtd_imagens_teste

# ARRAY COM TODOS OS MEUS INDICES EMBARALHADOSVI QUE É BOM TER SEED PRA PODER SER REPRODUZIDO DEPOIS)
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
