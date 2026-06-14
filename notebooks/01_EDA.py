# %% [markdown]
# # DiabetesScan — Notebook 01: Análise Exploratória de Dados (EDA)
# **Projeto A3 — IA Aplicada à Saúde — USJT 2026**
#
# Este notebook realiza a análise exploratória completa do dataset
# Pima Indians Diabetes antes de qualquer pré-processamento.

# %% Imports
import sys, os
sys.path.insert(0, os.path.join("..", "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import ZERO_COLS

sns.set_theme(style="whitegrid", palette="teal")
plt.rcParams["figure.dpi"] = 120

# %% [markdown]
# ## 1. Carregamento dos dados

# %%
df = pd.read_csv("../data/diabetes.csv")
print(f"Shape: {df.shape}")
df.head()

# %% [markdown]
# ## 2. Informações gerais

# %%
df.info()

# %%
df.describe().round(2)

# %% [markdown]
# ## 3. Distribuição da variável target (Outcome)

# %%
fig, ax = plt.subplots(figsize=(5, 3.5))
counts = df["Outcome"].value_counts()
bars = ax.bar(["Sem Diabetes (0)", "Com Diabetes (1)"],
              counts.values, color=["#028090", "#E74C3C"])
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f"{val}\n({val/len(df)*100:.1f}%)", ha="center", fontsize=11)
ax.set_title("Distribuição das Classes", fontsize=13, fontweight="bold")
ax.set_ylabel("Nº de Pacientes")
plt.tight_layout()
plt.savefig("../figures/eda_class_distribution.png", dpi=150)
plt.show()

# %% [markdown]
# **Observação:** O dataset está moderadamente desbalanceado (65% vs 35%).
# Será aplicado SMOTE no pipeline de treino.

# %% [markdown]
# ## 4. Zeros biologicamente impossíveis

# %%
zeros = pd.DataFrame({
    "Feature": ZERO_COLS,
    "Zeros": [df[col].eq(0).sum() for col in ZERO_COLS],
    "Percentual (%)": [(df[col].eq(0).sum() / len(df) * 100).round(1) for col in ZERO_COLS]
})
print(zeros.to_string(index=False))

# %% [markdown]
# Colunas como Insulin (48.7%) e SkinThickness (29.6%) têm muitos zeros
# que são biologicamente impossíveis e serão imputados pela mediana do grupo.

# %% [markdown]
# ## 5. Distribuições das features

# %%
fig, axes = plt.subplots(3, 3, figsize=(12, 9))
axes = axes.flatten()
for i, col in enumerate(df.columns[:-1]):
    axes[i].hist(df[df["Outcome"]==0][col], bins=25, alpha=0.6,
                 color="#028090", label="Sem Diabetes")
    axes[i].hist(df[df["Outcome"]==1][col], bins=25, alpha=0.6,
                 color="#E74C3C", label="Com Diabetes")
    axes[i].set_title(col, fontweight="bold")
    axes[i].legend(fontsize=7)
axes[-1].set_visible(False)
plt.suptitle("Distribuição das Features por Classe", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../figures/eda_distributions.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Matriz de correlação

# %%
fig, ax = plt.subplots(figsize=(9, 7))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, ax=ax, linewidths=0.5)
ax.set_title("Matriz de Correlação", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../figures/eda_correlation.png", dpi=150)
plt.show()

# Maiores correlações com Outcome
print("\nCorrelações com Outcome (descendente):")
print(corr["Outcome"].drop("Outcome").sort_values(ascending=False).round(3))

# %% [markdown]
# ## 7. Boxplots para detecção de outliers

# %%
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
axes = axes.flatten()
for i, col in enumerate(df.columns[:-1]):
    df.boxplot(column=col, by="Outcome", ax=axes[i],
               boxprops=dict(color="#028090"),
               medianprops=dict(color="#E74C3C", linewidth=2))
    axes[i].set_title(col, fontweight="bold")
    axes[i].set_xlabel("")
plt.suptitle("Boxplots por Classe (0 = Sem Diabetes, 1 = Com Diabetes)", fontsize=12)
plt.tight_layout()
plt.savefig("../figures/eda_boxplots.png", dpi=150)
plt.show()

# %% [markdown]
# ## 8. Pairplot das features mais correlacionadas

# %%
top_features = ["Glucose", "BMI", "Age", "DiabetesPedigreeFunction", "Outcome"]
g = sns.pairplot(df[top_features], hue="Outcome", diag_kind="kde",
                 palette={0: "#028090", 1: "#E74C3C"},
                 plot_kws={"alpha": 0.4})
g.fig.suptitle("Pairplot — Top Features vs Outcome", y=1.02, fontsize=13, fontweight="bold")
plt.savefig("../figures/eda_pairplot.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Sumário dos insights
#
# | Insight | Ação |
# |---------|------|
# | Zeros impossíveis em 5 colunas | Imputar pela mediana do grupo |
# | Desbalanceamento 65/35 | Aplicar SMOTE no treino |
# | Glucose maior correlação (0.47) | Feature mais importante esperada |
# | Outliers extremos em Insulin | Capping no percentil 99 |
# | Distribuições assimétricas | StandardScaler para normalização |
