# 🩺 DiabetesScan
### Predição de Risco de Diabetes Tipo 2 com Machine Learning

---

## 📋 Descrição

Sistema inteligente de triagem que prediz o risco de Diabetes Tipo 2 a partir de dados clínicos simples. Implementa três algoritmos de Machine Learning (Regressão Logística, Random Forest e XGBoost), com explicabilidade via SHAP Values e interface interativa em Streamlit.

**Melhor modelo:** XGBoost — ROC-AUC **0.913**

---

## 🗂️ Estrutura do Projeto

```
diabetes_scan/
├── data/
│   └── diabetes.csv              # Dataset Pima Indians Diabetes
├── notebooks/
│   ├── 01_EDA.py                 # Análise Exploratória de Dados
│   └── 02_modeling.py            # Modelagem e Avaliação
├── src/
│   ├── preprocessing.py          # Limpeza, FE, split, SMOTE
│   ├── models.py                 # Treinamento + GridSearchCV
│   ├── evaluation.py             # Métricas, confusão, ROC
│   └── explainability.py         # Feature Importance + SHAP
├── models/                       # Modelos treinados (gerado pelo train.py)
├── figures/                      # Gráficos gerados (gerado pelo train.py)
├── app.py                        # Interface Streamlit
├── train.py                      # Script de treinamento completo
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/aandrxy/diabetes-scan.git
cd diabetes-scan

# 2. Crie o ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# 3. Instale as dependências
pip install -r requirements.txt
```

---

## 📊 Dataset

**Pima Indians Diabetes Database**  
Fonte: UCI ML Repository / Kaggle  
Link: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database  

- 768 registros | 8 features | 1 target (Outcome)
- Coloque o arquivo `diabetes.csv` na pasta `data/`

---

## 🚀 Como usar

### 1. Treinar os modelos
```bash
python train.py
```
Isso executa o pipeline completo e salva os modelos em `/models` e figuras em `/figures`.

### 2. Executar a interface
```bash
streamlit run app.py
```
Acesse `http://localhost:8501` no navegador.

### 3. Executar os notebooks
```bash
# Converta os .py para .ipynb (opcional)
pip install jupytext
jupytext --to notebook notebooks/01_EDA.py
jupyter notebook notebooks/01_EDA.ipynb
```
Ou execute diretamente como scripts Python:
```bash
cd diabetes_scan
python notebooks/01_EDA.py
python notebooks/02_modeling.py
```

---

## 🤖 Modelos Implementados

| Modelo | Hiperparâmetros Otimizados | ROC-AUC |
|--------|---------------------------|---------|
| Regressão Logística | C=1.0, solver=liblinear | 0.831 |
| Random Forest | n_estimators=200, max_depth=10 | 0.877 |
| **XGBoost** ⭐ | n_estimators=300, lr=0.05, max_depth=6 | **0.913** |

Todos otimizados com **GridSearchCV + 5-Fold Cross Validation**.

---

## 🔬 Pipeline de Dados

1. **Imputação** — zeros biologicamente impossíveis → mediana por grupo
2. **Outliers** — capping no percentil 99
3. **Feature Engineering** — BMI_category, Age_group, Glucose_category
4. **Normalização** — StandardScaler (fit só no treino)
5. **Split** — 80% treino / 20% teste, estratificado
6. **Balanceamento** — SMOTE no conjunto de treino

---

## ⚖️ Aspectos Éticos

- Dataset original de mulheres Pima (EUA, 1988) — pode não generalizar para a população brasileira diversa
- O sistema é uma ferramenta de **apoio à triagem**, nunca substitui o médico
- Dados inseridos são processados localmente — conformidade com **LGPD** (Art. 11)
- Vieses de gênero e étnico devem ser avaliados antes de qualquer uso clínico real

---

## 📚 Referências

- Smith, J.W. et al. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proceedings of SCAMC*.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD'16*.
- Lundberg, S.M., & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
- LGPD — Lei nº 13.709/2018. Disponível em: https://www.planalto.gov.br
