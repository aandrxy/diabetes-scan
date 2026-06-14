"""
preprocessing.py
----------------
Carregamento, limpeza, engenharia de atributos e split treino/teste
para o dataset Pima Indians Diabetes.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

# Colunas com zeros biologicamente impossíveis
ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def load_data(path: str = "data/diabetes.csv") -> pd.DataFrame:
    """Carrega o CSV do dataset Pima Indians Diabetes."""
    df = pd.read_csv(path)
    return df


def replace_impossible_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Substitui zeros impossíveis pela mediana calculada por grupo de Outcome.
    Evita data leakage: mediana calculada antes do split é aceitável para EDA,
    mas no pipeline de treino é recalculada apenas no conjunto de treino.
    """
    df = df.copy()
    for col in ZERO_COLS:
        medians = df.groupby("Outcome")[col].median()
        mask = df[col] == 0
        df.loc[mask, col] = df.loc[mask, "Outcome"].map(medians)
    return df


def cap_outliers(df: pd.DataFrame, percentile: float = 0.99) -> pd.DataFrame:
    """Capping no percentil informado para reduzir efeito de outliers extremos."""
    df = df.copy()
    for col in df.columns.drop("Outcome"):
        cap = df[col].quantile(percentile)
        df[col] = df[col].clip(upper=cap)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Cria atributos derivados clinicamente relevantes."""
    df = df.copy()
    # Faixa de IMC (OMS)
    df["BMI_category"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=[0, 1, 2, 3]   # baixo, normal, sobrepeso, obeso
    ).astype(float)

    # Faixa etária
    df["Age_group"] = pd.cut(
        df["Age"],
        bins=[0, 30, 45, 60, 120],
        labels=[0, 1, 2, 3]
    ).astype(float)

    # Categoria de glicemia (ADA)
    df["Glucose_category"] = pd.cut(
        df["Glucose"],
        bins=[0, 99, 125, 500],
        labels=[0, 1, 2]   # normal, pré-diabético, diabético
    ).astype(float)

    return df


def build_pipeline(path: str = "data/diabetes.csv",
                   test_size: float = 0.2,
                   random_state: int = 42,
                   apply_smote: bool = True):
    """
    Pipeline completo:
    1. Carrega dados
    2. Trata zeros e outliers
    3. Engenharia de atributos
    4. Split treino/teste estratificado
    5. Normalização (StandardScaler fitado apenas no treino)
    6. SMOTE no conjunto de treino

    Retorna: X_train, X_test, y_train, y_test, scaler, feature_names
    """
    df = load_data(path)
    df = replace_impossible_zeros(df)
    df = cap_outliers(df)
    df = feature_engineering(df)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Normalização — fit apenas no treino
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    if apply_smote:
        sm = SMOTE(random_state=random_state)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, scaler, feature_names


def save_scaler(scaler, path: str = "models/scaler.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: str = "models/scaler.pkl"):
    return joblib.load(path)
