import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


def avaliar_modelo(modelo, dataloader, dispositivo, nomes_classes, nome_arquivo_saida):
    modelo.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for imagens, rotulos in dataloader:
            imagens = imagens.to(dispositivo)
            rotulos = rotulos.to(dispositivo)

            saidas = modelo(imagens)
            _, predicoes = torch.max(saidas, 1)

            y_true.extend(rotulos.cpu().numpy())
            y_pred.extend(predicoes.cpu().numpy())

    # Conversão para numpy array
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ====== Cálculos ======
    matriz_confusao = confusion_matrix(y_true, y_pred)
    acuracia = accuracy_score(y_true, y_pred)
    precisao = precision_score(
        y_true, y_pred, average="macro", zero_division=0)
    revocacao = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # ====== Relatório detalhado ======
    relatorio = classification_report(
        y_true, y_pred, target_names=nomes_classes, zero_division=0)

    # ====== Salvando no arquivo ======
    with open(nome_arquivo_saida, "w", encoding="utf-8") as f:
        f.write("==== AVALIAÇÃO DO MODELO ====\n\n")
        f.write(f"Acurácia: {acuracia:.4f}\n")
        f.write(f"Precisão (macro): {precisao:.4f}\n")
        f.write(f"Revocação (macro): {revocacao:.4f}\n")
        f.write(f"F1-score (macro): {f1:.4f}\n\n")
        f.write("==== Relatório por classe ====\n")
        f.write(relatorio)
        f.write("\n==== Matriz de Confusão ====\n")
        f.write(np.array2string(matriz_confusao))

    # ====== Exibir matriz de confusão (opcional) ======
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_confusao, annot=True, fmt="d", cmap="Blues",
                xticklabels=nomes_classes, yticklabels=nomes_classes)
    plt.title("Matriz de Confusão")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(nome_arquivo_saida.replace(".txt", "_matriz_confusao.png"))
    plt.close()

    print(f"✅ Avaliação salva em: {nome_arquivo_saida}")
