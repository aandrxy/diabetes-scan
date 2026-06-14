"""
models.py
---------
Treinamento e otimização dos três modelos com GridSearchCV (5-Fold CV).
Suporte a salvar/carregar modelos treinados.
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier


# ── Configurações de hiperparâmetros ──────────────────────────────────────────

LR_PARAMS = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "solver": ["liblinear"],
    "max_iter": [300],
}

RF_PARAMS = {
    "n_estimators": [100, 200],
    "max_depth": [6, 10, None],
    "min_samples_leaf": [2, 4],
}

XGB_PARAMS = {
    "n_estimators": [200, 300],
    "learning_rate": [0.05, 0.1],
    "max_depth": [4, 6],
    "colsample_bytree": [0.8, 1.0],
    "subsample": [0.8],
}


def _grid_search(estimator, param_grid, X_train, y_train,
                 cv: int = 5, scoring: str = "roc_auc"):
    """Executa GridSearchCV e retorna o melhor estimador."""
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    gs = GridSearchCV(
        estimator, param_grid,
        cv=kf, scoring=scoring,
        n_jobs=-1, verbose=0, refit=True
    )
    gs.fit(X_train, y_train)
    print(f"  Melhores params: {gs.best_params_}")
    print(f"  Melhor {scoring} (CV): {gs.best_score_:.4f}")
    return gs.best_estimator_


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    print("\n[1/3] Treinando Regressão Logística...")
    base = LogisticRegression(random_state=42)
    return _grid_search(base, LR_PARAMS, X_train, y_train)


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    print("\n[2/3] Treinando Random Forest...")
    base = RandomForestClassifier(random_state=42)
    return _grid_search(base, RF_PARAMS, X_train, y_train)


def train_xgboost(X_train, y_train) -> XGBClassifier:
    print("\n[3/3] Treinando XGBoost...")
    base = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    return _grid_search(base, XGB_PARAMS, X_train, y_train)


def train_all(X_train, y_train) -> dict:
    """Treina os três modelos e retorna dicionário com os estimadores."""
    models = {
        "Regressão Logística": train_logistic_regression(X_train, y_train),
        "Random Forest":       train_random_forest(X_train, y_train),
        "XGBoost":             train_xgboost(X_train, y_train),
    }
    return models


def save_models(models: dict, folder: str = "models"):
    os.makedirs(folder, exist_ok=True)
    for name, model in models.items():
        fname = name.lower().replace(" ", "_").replace("ã", "a").replace("ç", "c") + ".pkl"
        path = os.path.join(folder, fname)
        joblib.dump(model, path)
        print(f"  Salvo: {path}")


def load_models(folder: str = "models") -> dict:
    mapping = {
        "Regressão Logística": "regressao_logistica.pkl",
        "Random Forest":       "random_forest.pkl",
        "XGBoost":             "xgboost.pkl",
    }
    models = {}
    for name, fname in mapping.items():
        path = os.path.join(folder, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models
