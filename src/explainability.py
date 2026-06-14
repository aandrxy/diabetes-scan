"""
explainability.py
-----------------
Feature Importance (Random Forest / XGBoost) e SHAP Values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def plot_feature_importance(model, feature_names: list,
                            model_name: str = "Modelo",
                            save_path: str = None):
    """
    Plota importância de features nativa do modelo
    (disponível para Random Forest e XGBoost).
    """
    if not hasattr(model, "feature_importances_"):
        print(f"  {model_name} não possui feature_importances_. Pulando.")
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)

    plt.figure(figsize=(8, 5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(importances)))
    plt.barh([feature_names[i] for i in idx], importances[idx], color=colors)
    plt.xlabel("Importância (Gini / Gain)")
    plt.title(f"Feature Importance — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figura salva: {save_path}")
    plt.show()


def compute_shap_values(model, X, feature_names: list,
                        model_name: str = "XGBoost"):
    """
    Calcula SHAP values usando TreeExplainer (para RF e XGBoost)
    ou LinearExplainer (para Regressão Logística).
    Retorna explainer e shap_values.
    """
    if hasattr(model, "feature_importances_"):
        # Tree-based
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # Para classificadores binários, shap_values pode ser lista [neg, pos]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        # Linear
        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer.shap_values(X)

    return explainer, shap_values


def plot_shap_summary(model, X, feature_names: list,
                      model_name: str = "Modelo",
                      save_path: str = None):
    """Beeswarm plot dos SHAP values (visão global)."""
    _, shap_values = compute_shap_values(model, X, feature_names, model_name)

    plt.figure(figsize=(9, 5))
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        show=False,
        plot_type="dot"
    )
    plt.title(f"SHAP Summary — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figura salva: {save_path}")
    plt.show()


def plot_shap_bar(model, X, feature_names: list,
                  model_name: str = "Modelo",
                  save_path: str = None):
    """Bar plot de importância global média |SHAP|."""
    _, shap_values = compute_shap_values(model, X, feature_names, model_name)
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)

    plt.figure(figsize=(8, 5))
    colors = plt.cm.GnBu(np.linspace(0.4, 0.9, len(mean_abs)))
    plt.barh([feature_names[i] for i in idx], mean_abs[idx], color=colors)
    plt.xlabel("Importância Média |SHAP|")
    plt.title(f"SHAP Global — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figura salva: {save_path}")
    plt.show()


def explain_single_prediction(model, X_single: np.ndarray,
                               feature_names: list,
                               model_name: str = "XGBoost"):
    """
    Retorna SHAP values para uma única predição.
    Útil para exibir na interface Streamlit.
    """
    _, shap_values = compute_shap_values(model, X_single, feature_names, model_name)
    sv = shap_values[0] if shap_values.ndim == 2 else shap_values
    result = pd.Series(sv, index=feature_names).sort_values(key=abs, ascending=False)
    return result
