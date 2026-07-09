import pandas as pd

# ETAPA 1: Carregamento dos dados
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)
print(df.head())

# ETAPA 2: Análise da variável-alvo (checar desbalanceamento)
print(df["Class"].value_counts(normalize=True))

# ETAPA 3: Feature engineering
import numpy as np

# Reduz assimetria da coluna Amount
df["Amount_Log"] = np.log1p(df["Amount"])

from sklearn.preprocessing import StandardScaler

# Padroniza escala (média 0, desvio padrão 1)
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

# ETAPA 4: Separação treino/teste
from sklearn.model_selection import train_test_split

X = df.drop("Class", axis=1)
y = df["Class"]

# stratify=y mantém a proporção de fraude igual no treino e no teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# ETAPA 5: Treinamento do modelo (baseline)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ETAPA 6: Avaliação do modelo
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# ETAPA 7: Curva ROC e AUC
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

print("AUC:", roc_auc_score(y_test, y_probs))

# ETAPA 8: Curva Precision-Recall (mais confiável em dados desbalanceados)
from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, y_probs)

plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

# ETAPA 9: Tratando o desbalanceamento (Undersampling)
fraudes = df[df["Class"] == 1]
normais = df[df["Class"] == 0].sample(len(fraudes), random_state=42)
df_under = pd.concat([fraudes, normais])

# Retreinando com os dados balanceados por undersampling
X_under = df_under.drop("Class", axis=1)
y_under = df_under["Class"]

model_under = LogisticRegression(max_iter=1000)
model_under.fit(X_under, y_under)
y_pred_under = model_under.predict(X_test)

print("=== Resultado com Undersampling ===")
print(classification_report(y_test, y_pred_under))

# ETAPA 10: Tratando o desbalanceamento (Oversampling com SMOTE)
from imblearn.over_sampling import SMOTE

# CORRIGIDO: random_state para reprodutibilidade
smote = SMOTE(random_state=42)

# CORRIGIDO: usa X_train/y_train (não X, y) para evitar vazamento de dados (data leakage)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Retreinando com os dados balanceados por SMOTE
model_smote = LogisticRegression(max_iter=1000)
model_smote.fit(X_res, y_res)
y_pred_smote = model_smote.predict(X_test)

print("=== Resultado com SMOTE ===")
print(classification_report(y_test, y_pred_smote))

# ETAPA 11: Random Forest com class_weight balanceado
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    class_weight="balanced",  # compensa o desbalanceamento sem precisar de SMOTE/undersampling
    n_jobs=-1,                # usa todos os núcleos do processador (treina mais rápido)
    random_state=42
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(classification_report(y_test, y_pred_rf))

# ETAPA 12: Pipeline (encadeia pré-processamento + modelo em um único objeto)
from sklearn.pipeline import Pipeline

# Pipeline evita erros manuais (ex: esquecer de escalar o teste) e organiza
# o fluxo: cada etapa roda em sequência, sempre na mesma ordem.
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# ETAPA 13: Ajustando o limiar de decisão manualmente
# Por padrão, predict() usa threshold = 0.5. Baixando para 0.3, o modelo
# passa a classificar como fraude com menos "certeza", aumentando o recall
# (pega mais fraudes) mas tende a reduzir a precision (mais falsos positivos).
threshold = 0.3

y_pred_custom = (y_probs > threshold).astype(int)

print(classification_report(y_test, y_pred_custom))

# ETAPA 14: XGBoost
from xgboost import XGBClassifier

xgb = XGBClassifier(
    scale_pos_weight=10,  # dá mais peso à classe minoritária (fraude), ajuda com desbalanceamento
    eval_metric="logloss"
    # obs: use_label_encoder foi removido (parâmetro descontinuado em versões recentes do XGBoost)
)

xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print(classification_report(y_test, y_pred_xgb))

# ETAPA 15: Importância das variáveis
import matplotlib.pyplot as plt

# feature_importances_ mostra o quanto cada coluna contribuiu para as
# decisões do modelo XGBoost (quanto maior, mais relevante para prever fraude)
importancias = xgb.feature_importances_

plt.bar(range(len(importancias)), importancias)
plt.title("Importância das variáveis")
plt.show()

# ETAPA 16: GridSearchCV (busca automática dos melhores hiperparâmetros)
from sklearn.model_selection import GridSearchCV

# Define as combinações de parâmetros que serão testadas
param_grid = {
    "max_depth": [3, 5],
    "n_estimators": [50, 100]
}

# GridSearchCV testa TODAS as combinações do param_grid usando validação
# cruzada (cv=3 = divide o treino em 3 partes e testa em cada uma),
# escolhendo a combinação que gera o melhor "recall" (métrica mais
# importante aqui, já que queremos detectar o máximo de fraudes possível)
grid = GridSearchCV(
    XGBClassifier(eval_metric="logloss"),
    param_grid,
    scoring="recall",
    cv=3
)

grid.fit(X_train, y_train)

print("Melhor modelo:", grid.best_params_)

# ETAPA 17: SHAP (explicabilidade do modelo)
import shap

# SHAP explica, para cada previsão, o quanto cada variável "empurrou"
# a decisão do modelo para fraude ou não-fraude. É mais detalhado que
# a importância de variáveis da Etapa 15, que só mostra uma média geral.
explainer = shap.Explainer(xgb)

# Calcula os valores SHAP só para as 100 primeiras linhas do teste
# (calcular para o dataset inteiro pode ser lento)
shap_values = explainer(X_test[:100])

# Gráfico de barras: mostra, em média, quais variáveis mais influenciaram
# as previsões do modelo nessas 100 amostras
shap.plots.bar(shap_values)
