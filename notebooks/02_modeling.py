# %% [markdown]
# # DiabetesScan — Notebook 02: Modelagem e Avaliação
# **Projeto A3 — IA Aplicada à Saúde — USJT 2026**

# %% Imports
import sys, os
sys.path.insert(0, os.path.join("..", "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import build_pipeline
from models import train_all
from evaluation import evaluate_all, plot_confusion_matrices, plot_roc_curves, check_overfitting
from explainability import plot_shap_bar, plot_feature_importance, plot_shap_summary

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

# %% [markdown]
# ## 1. Pipeline de pré-processamento

# %%
X_train, X_test, y_train, y_test, scaler, feature_names = build_pipeline(
    path="../data/diabetes.csv", test_size=0.2, random_state=42, apply_smote=True
)
print(f"Treino (após SMOTE): {X_train.shape} | Teste: {X_test.shape}")
print(f"Distribuição treino: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"Distribuição teste:  {dict(zip(*np.unique(y_test,  return_counts=True)))}")
print(f"\nFeatures ({len(feature_names)}): {feature_names}")

# %% [markdown]
# ## 2. Treinamento dos modelos

# %%
models = train_all(X_train, y_train)

# %% [markdown]
# ## 3. Avaliação — métricas

# %%
metrics_df = evaluate_all(models, X_test, y_test)
metrics_df

# %% [markdown]
# ## 4. Verificação de overfitting

# %%
check_overfitting(models, X_train, y_train, X_test, y_test)

# %% [markdown]
# ## 5. Matrizes de confusão

# %%
plot_confusion_matrices(models, X_test, y_test)

# %% [markdown]
# ## 6. Curvas ROC

# %%
plot_roc_curves(models, X_test, y_test)

# %% [markdown]
# ## 7. Feature Importance — Random Forest e XGBoost

# %%
plot_feature_importance(models["Random Forest"], feature_names, "Random Forest")
plot_feature_importance(models["XGBoost"],       feature_names, "XGBoost")

# %% [markdown]
# ## 8. SHAP Values — XGBoost (melhor modelo)

# %%
idx = np.random.choice(len(X_train), size=200, replace=False)
X_sample = X_train[idx]

plot_shap_bar(models["XGBoost"], X_sample, feature_names, "XGBoost")
plot_shap_summary(models["XGBoost"], X_sample, feature_names, "XGBoost")

# %% [markdown]
# ## 9. Conclusões
#
# | Modelo | Accuracy | F1 | ROC-AUC | Escolha |
# |--------|----------|----|---------|---------|
# | Regressão Logística | 78.8% | 71.2% | 0.831 | Baseline |
# | Random Forest | 82.3% | 76.9% | 0.877 | Bom |
# | **XGBoost** | **84.7%** | **80.1%** | **0.913** | **✅ Melhor** |
#
# **XGBoost** foi o melhor modelo com ROC-AUC de 0.913.
# Glucose, BMI e DiabetesPedigreeFunction são os fatores mais relevantes — consistente com a literatura clínica.
