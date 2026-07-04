# Detectando Fraudes em Transações Bancárias

Projeto de detecção de fraudes em transações com cartão de crédito, desenvolvido durante o **Bootcamp Bradesco - GenAI da DIO**, sob orientação da professora **Isadora Garcia Ferrão**.

## Sobre o Dataset

O dataset utilizado é o [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), disponível no Kaggle. Ele contém transações financeiras realizadas com cartão de crédito europeus em setembro de 2013.

- **Time**: segundos decorridos entre a primeira transação e a transação atual
- **V1 a V28**: componentes PCA (técnica de privacidade)
- **Amount**: valor da transação
- **Class**: 0 = não fraudulenta, 1 = fraudulenta

> **Nota:** O dataset é altamente desbalanceado (~0,17% de fraudes).

## Pipeline

1. Carregamento e análise exploratória dos dados
2. Verificação do balanceamento das classes
3. Análise dos valores das transações (Amount)
4. Feature Engineering: transformação logarítmica (`log1p`) do Amount
5. Padronização com `StandardScaler`
6. Divisão treino/teste com estratificação
7. Treinamento com Regressão Logística
8. Avaliação com Classification Report, ROC-AUC e Precision-Recall Curve

## Métricas

Para datasets desbalanceados como este, as métricas mais relevantes são:

- **Recall**: taxa de detecção de fraudes
- **Precision**: assertividade nas classificações de fraude
- **ROC-AUC**: capacidade de distinguir classes
- **Precision-Recall AUC**: mais informativa que ROC em dados desbalanceados

## Como usar

```bash
pip install -r requirements.txt
python fraud_detection.py
```

## Resultados

O script gera gráficos de análise exploratória e curvas de avaliação do modelo:

- `class_balance.png` — distribuição entre fraudes e não fraudes
- `amount_analysis.png` — valores por classe
- `amount_log_transform.png` — efeito da transformação logarítmica
- `roc_curve.png` — curva ROC
- `precision_recall_curve.png` — curva Precision-Recall

## Tecnologias

- Python 3
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
