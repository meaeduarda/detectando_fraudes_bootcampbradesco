# Detecção de Fraudes em Transações Bancárias

[![Bootcamp DIO](https://img.shields.io/badge/Bootcamp-DIO-blue)](https://www.dio.me)
[![Bradesco](https://img.shields.io/badge/Parceria-Bradesco-red)](https://banco.bradesco)

Projeto desenvolvido durante o **Bootcamp Bradesco - GenAI** da **Digital Innovation One (DIO)** sob a orientação da professora **Isadora Garcia Ferrão**.

## Sobre o Projeto

Pipeline completo de análise exploratória, feature engineering e modelagem para detecção de transações fraudulentas utilizando **Regressão Logística**. O dataset utilizado contém transações reais de cartão de crédito anonimizadas via PCA.

## Pipeline

### 1. Carregamento dos Dados
- Dataset público de detecção de fraudes (284.807 transações)
- 30 features: `Time`, `V1` a `V28` (componentes PCA), `Amount` e `Class`

### 2. Análise Exploratória
- Verificação de balanceamento das classes
- Dataset altamente desbalanceado: ~0,17% de fraudes
- Geração do gráfico `class_balance.png`

### 3. Análise dos Valores (Amount)
- Distribuição dos montantes das transações
- Comparação entre transações fraudulentas e legítimas
- Geração do gráfico `amount_analysis.png`

### 4. Feature Engineering
- Criação da feature `Amount_log` com `np.log1p` para reduzir assimetria dos valores financeiros
- Geração do gráfico `amount_log_transform.png`

### 5. Padronização com StandardScaler
- Padronização das colunas `Time` e `Amount_log` para média 0 e desvio padrão 1

### 6. Divisão Treino/Teste
- Separação estratificada: 70% treino / 30% teste
- Mantém a proporção original de fraudes em ambos os conjuntos

### 7. Treinamento do Modelo
- Algoritmo: Regressão Logística
- Hiperparâmetros: `max_iter=1000`

### 8. Avaliação
- **Classification Report**: Precision, Recall e F1-Score para ambas as classes
- **ROC-AUC Score**: capacidade do modelo de distinguir classes
- **Curva ROC**: gráfico `roc_curve.png`
- **Precision-Recall Curve**: métrica mais informativa para dados desbalanceados — gráfico `precision_recall_curve.png`

### 9. Interpretação dos Resultados
Para detecção de fraudes, as métricas mais importantes são:
- **Recall** — quantas fraudes reais foram capturadas
- **Precision** — das classificadas como fraude, quantas realmente são
- **ROC-AUC** — capacidade geral de discriminação
- **Precision-Recall AUC** — mais robusta para classes desbalanceadas

## Tecnologias Utilizadas

| Ferramenta | Versão |
|-----------|--------|
| Python | 3.x |
| pandas | >= 1.3.0 |
| numpy | >= 1.21.0 |
| scikit-learn | >= 1.0.0 |
| matplotlib | >= 3.4.0 |
| seaborn | >= 0.11.0 |

## Como Executar

```bash
# Instalar dependências
pip install -r requirements.txt

# Executar o pipeline completo
python fraud_detection.py
```

## Resultados

O script gera automaticamente 5 gráficos analíticos:

| Arquivo | Descrição |
|---------|-----------|
| `class_balance.png` | Balanceamento entre classes |
| `amount_analysis.png` | Distribuição dos valores por classe |
| `amount_log_transform.png` | Efeito da transformação logarítmica |
| `roc_curve.png` | Curva ROC com AUC |
| `precision_recall_curve.png` | Curva Precision-Recall com AUC |

## Possíveis Melhorias

- Técnicas de balanceamento: SMOTE, undersampling, `class_weight='balanced'`
- Modelos mais robustos: Random Forest, XGBoost
- Ajuste de threshold na curva Precision-Recall
- Validação cruzada para otimização de hiperparâmetros

---

*Projeto educacional desenvolvido no Bootcamp Bradesco GenAI — DIO*
