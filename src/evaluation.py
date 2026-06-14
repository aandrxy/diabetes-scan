"""
evaluation.py
-------------
Avaliação dos modelos com métricas completas, matriz de confusão e curva ROC.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)


def evaluate_model(model, X_test, y_test, model_name: str = "") -> dict:
    """Calcula e exibe todas as métricas para um modelo."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Modelo":    model_name,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall":    round(recall_score(y_test, y_pred), 4),
        "F1-Score":  round(f1_score(y_test, y_pred), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_proba), 4),
    }

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    for k, v in metrics.items():
        if k != "Modelo":
            print(f"  {k:<12}: {v}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Sem Diabetes','Com Diabetes'])}")

    return metrics


def evaluate_all(models: dict, X_test, y_test) -> pd.DataFrame:
    """Avalia todos os modelos e retorna DataFrame comparativo."""
    rows = []
    for name, model in models.items():
        row = evaluate_model(model, X_test, y_test, name)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Modelo")
    return df


def plot_confusion_matrices(models: dict, X_test, y_test,
                            save_path: str = None):
    """Plota matrizes de confusão lado a lado."""
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Negativo", "Positivo"],
                    yticklabels=["Negativo", "Positivo"])
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")

    plt.suptitle("Matrizes de Confusão — Conjunto de Teste", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figura salva: {save_path}")
    plt.show()


def plot_roc_curves(models: dict, X_test, y_test,
                    save_path: str = None):
    """Plota curvas ROC de todos os modelos sobrepostas."""
    colors = ["#028090", "#00A896", "#F39C12"]
    plt.figure(figsize=(7, 5))

    for (name, model), color in zip(models.items(), colors):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color, lw=2)

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    plt.xlabel("Taxa de Falso Positivo (FPR)")
    plt.ylabel("Taxa de Verdadeiro Positivo (TPR)")
    plt.title("Curvas ROC — Comparação de Modelos", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figura salva: {save_path}")
    plt.show()


def check_overfitting(models: dict, X_train, y_train, X_test, y_test):
    """Compara acurácia de treino vs teste para detectar overfitting."""
    print("\n── Verificação de Overfitting ──────────────────────")
    for name, model in models.items():
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc  = accuracy_score(y_test,  model.predict(X_test))
        gap = train_acc - test_acc
        status = "✅ OK" if gap < 0.05 else "⚠️  Possível overfitting"
        print(f"  {name:<25} treino={train_acc:.3f}  teste={test_acc:.3f}  gap={gap:.3f}  {status}")
