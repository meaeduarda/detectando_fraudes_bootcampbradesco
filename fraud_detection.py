"""
Detecção de Fraudes em Transações Bancárias
Bootcamp Bradesco - GenAI DIO
Professora: Isadora Garcia Ferrão

Pipeline completo de análise e detecção de fraudes usando Regressão Logística.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
)

sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (10, 6)

# 1. CARREGAMENTO DOS DADOS
URL = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"

df = pd.read_csv(URL)

print("=" * 60)
print("1. VISÃO INICIAL DOS DADOS")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nPrimeiras linhas:\n{df.head()}")
print(f"\nInformações do dataset:\n{df.info()}")
print(f"\nValores nulos:\n{df.isnull().sum().sum()}")


# 2. ANÁLISE EXPLORATÓRIA - BALANCEAMENTO DAS CLASSES

print("\n" + "=" * 60)
print("2. BALANCEAMENTO DAS CLASSES")
print("=" * 60)

class_counts = df["Class"].value_counts()
class_proportions = df["Class"].value_counts(normalize=True)

print(f"Contagem:\n{class_counts}")
print(f"\nProporção:\n{class_proportions}")

# Visualização do desbalanceamento
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(
    ["Não Fraude (0)", "Fraude (1)"],
    class_counts.values,
    color=["royalblue", "crimson"],
)
axes[0].set_title("Contagem de Transações")
axes[0].set_ylabel("Quantidade")

axes[1].bar(
    ["Não Fraude (0)", "Fraude (1)"],
    class_proportions.values * 100,
    color=["royalblue", "crimson"],
)
axes[1].set_title("Proporção (%)")
axes[1].set_ylabel("Percentual")
plt.tight_layout()
plt.savefig("class_balance.png")
plt.show()

print("\n>>> Dataset altamente desbalanceado!")
print(">>> Acurácia isolada é enganosa - usaremos Precision, Recall e ROC-AUC.")


# 3. ANÁLISE DOS VALORES DAS TRANSAÇÕES

print("\n" + "=" * 60)
print("3. ANÁLISE DOS VALORES (AMOUNT)")
print("=" * 60)

print(f"Estatísticas do Amount:\n{df['Amount'].describe()}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["Amount"], bins=50, color="steelblue", edgecolor="black")
axes[0].set_title("Distribuição do Amount (original)")
axes[0].set_xlabel("Valor")
axes[0].set_ylabel("Frequência")

axes[1].hist(
    df[df["Class"] == 0]["Amount"],
    bins=50,
    alpha=0.6,
    label="Não Fraude",
    color="royalblue",
)
axes[1].hist(
    df[df["Class"] == 1]["Amount"],
    bins=50,
    alpha=0.8,
    label="Fraude",
    color="crimson",
)
axes[1].set_title("Amount por Classe")
axes[1].set_xlabel("Valor")
axes[1].set_ylabel("Frequência")
axes[1].legend()
plt.tight_layout()
plt.savefig("amount_analysis.png")
plt.show()


# 4. FEATURE ENGINEERING

print("\n" + "=" * 60)
print("4. FEATURE ENGINEERING")
print("=" * 60)

# Aplicando log1p para reduzir a escala dos valores financeiros
df["Amount_log"] = np.log1p(df["Amount"])

print(">>> Coluna 'Amount_log' criada com np.log1p(df['Amount'])")
print(">>> log1p = log(1 + x) - útil para dados financeiros com escalas muito diferentes.")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["Amount"], bins=50, color="steelblue", edgecolor="black")
axes[0].set_title("Amount Original")
axes[1].hist(df["Amount_log"], bins=50, color="seagreen", edgecolor="black")
axes[1].set_title("Amount_log (log1p)")
plt.tight_layout()
plt.savefig("amount_log_transform.png")
plt.show()


# 5. PADRONIZAÇÃO DOS DADOS (STANDARD SCALER)

print("\n" + "=" * 60)
print("5. PADRONIZAÇÃO COM STANDARDSCALER")
print("=" * 60)

scaler = StandardScaler()

# V1-V28 já são componentes PCA (padronizadas), mas Time e Amount_log precisam de escala
cols_to_scale = ["Time", "Amount_log"]
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

print(f">>> Colunas padronizadas: {cols_to_scale}")
print(f">>> Média após scaling:\n{df[cols_to_scale].mean()}")
print(f">>> Desvio padrão após scaling:\n{df[cols_to_scale].std()}")


# 6. TREINO / TESTE SPLIT

print("\n" + "=" * 60)
print("6. DIVISÃO TREINO/TESTE")
print("=" * 60)

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
print(f"Proporção de fraudes no treino: {y_train.mean():.4f}")
print(f"Proporção de fraudes no teste:  {y_test.mean():.4f}")


# 7. TREINAMENTO DO MODELO - REGRESSÃO LOGÍSTICA

print("\n" + "=" * 60)
print("7. TREINAMENTO - REGRESSÃO LOGÍSTICA")
print("=" * 60)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

print(">>> Modelo treinado com sucesso!")
print(f">>> Número de iterações: {model.n_iter_[0]}")


# 8. AVALIAÇÃO DO MODELO

print("\n" + "=" * 60)
print("8. AVALIAÇÃO DO MODELO")
print("=" * 60)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["Não Fraude", "Fraude"]))

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_proba)
print(f"\n--- ROC-AUC Score: {roc_auc:.4f} ---")

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)

plt.figure()
plt.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.4f})", linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall")
plt.legend(loc="lower left")
plt.savefig("precision_recall_curve.png")
plt.show()

print(f"\n--- Precision-Recall AUC: {pr_auc:.4f} ---")


# 9. INTERPRETAÇÃO DOS RESULTADOS

print("\n" + "=" * 60)
print("9. INTERPRETAÇÃO DOS RESULTADOS")
print("=" * 60)

print("""
>>> Para detecção de fraudes (classes desbalanceadas), as métricas mais importantes são:
    - Recall (Taxa de detecção de fraudes): quantas fraudes reais foram capturadas
    - Precision: das transações classificadas como fraude, quantas são realmente fraude
    - ROC-AUC: capacidade geral do modelo de distinguir entre classes
    - Precision-Recall AUC: mais informativa que ROC quando há desbalanceamento severo

>>> Como melhorar:
    - Técnicas de balanceamento: SMOTE, undersampling, class_weight='balanced'
    - Outros modelos: Random Forest, XGBoost
    - Ajuste de threshold na curva Precision-Recall
""")


