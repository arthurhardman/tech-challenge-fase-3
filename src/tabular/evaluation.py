"""
evaluation.py
-------------
Métricas, visualizações e interpretabilidade dos modelos de classificação.

Funcionalidades:
  - Relatório de métricas (accuracy, precision, recall, F1, ROC-AUC)
  - Matriz de confusão
  - Curvas ROC e Precision-Recall
  - Feature Importance (modelos baseados em árvores)
  - SHAP values para explicabilidade
  - Comparativo entre modelos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import shap
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

matplotlib.use("Agg")  # Backend não-interativo para salvar figuras sem display

FIGURES_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def calcular_metricas(
    nome: str,
    modelo,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Calcula as principais métricas de classificação para um modelo.

    Retorna
    -------
    dict com accuracy, precision, recall, f1, roc_auc
    """
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else None

    metricas = {
        "modelo": nome,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
    }

    print(f"\n[{nome}]")
    print(classification_report(y_test, y_pred, target_names=["Cura (0)", "Óbito (1)"]))
    if metricas["roc_auc"]:
        print(f"ROC-AUC: {metricas['roc_auc']:.4f}")

    return metricas


def comparar_modelos(resultados: list[dict]) -> pd.DataFrame:
    """
    Exibe tabela comparativa de métricas entre modelos.

    Parâmetros
    ----------
    resultados : list de dicts retornados por calcular_metricas()

    Retorna
    -------
    pd.DataFrame ordenado por F1-score decrescente
    """
    df = pd.DataFrame(resultados).set_index("modelo")
    df = df.sort_values("f1", ascending=False)
    df = df.round(4)
    print("\nComparativo de modelos:")
    print(df.to_string())
    return df


def plotar_matriz_confusao(nome: str, modelo, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Plota e salva a matriz de confusão do modelo.
    """
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Cura (0)", "Óbito (1)"],
        yticklabels=["Cura (0)", "Óbito (1)"],
    )
    ax.set_xlabel("Predição")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusão — {nome}")
    plt.tight_layout()

    caminho = FIGURES_DIR / f"confusion_matrix_{nome}.png"
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"Figura salva: {caminho}")


def plotar_curvas_roc(modelos: dict, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Plota e salva curvas ROC para todos os modelos em um único gráfico.

    Parâmetros
    ----------
    modelos : dict { nome: estimator }
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Aleatório (AUC=0.50)")

    for nome, modelo in modelos.items():
        if hasattr(modelo, "predict_proba"):
            y_prob = modelo.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, label=f"{nome} (AUC={auc:.3f})")

    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title("Curvas ROC — Comparativo de Modelos")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    caminho = FIGURES_DIR / "roc_curves.png"
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"Figura salva: {caminho}")


def plotar_curvas_precision_recall(modelos: dict, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Plota e salva curvas Precision-Recall para todos os modelos.
    Importante em datasets desbalanceados como alternativa ao ROC.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for nome, modelo in modelos.items():
        if hasattr(modelo, "predict_proba"):
            y_prob = modelo.predict_proba(X_test)[:, 1]
            prec, rec, _ = precision_recall_curve(y_test, y_prob)
            ax.plot(rec, prec, label=nome)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall — Comparativo de Modelos")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    caminho = FIGURES_DIR / "precision_recall_curves.png"
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"Figura salva: {caminho}")


def plotar_feature_importance(nome: str, modelo, feature_names: list, top_n: int = 20) -> None:
    """
    Plota feature importance para modelos baseados em árvores (RF, XGBoost, DT).
    Exibe as top_n features mais importantes.
    """
    if not hasattr(modelo, "feature_importances_"):
        print(f"[{nome}] Modelo não suporta feature_importances_. Pulando.")
        return

    importances = modelo.feature_importances_
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, top_n // 2)))
    sns.barplot(data=df_imp, x="importance", y="feature", palette="viridis", ax=ax)
    ax.set_title(f"Feature Importance (Top {top_n}) — {nome}")
    ax.set_xlabel("Importância")
    ax.set_ylabel("")
    plt.tight_layout()

    caminho = FIGURES_DIR / f"feature_importance_{nome}.png"
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"Figura salva: {caminho}")


def plotar_shap(nome: str, modelo, X_test: pd.DataFrame, n_samples: int = 500) -> None:
    """
    Gera e salva gráficos SHAP (summary plot) para interpretabilidade do modelo.

    Usa uma amostra de X_test para eficiência computacional.

    Parâmetros
    ----------
    nome : str
    modelo : estimator treinado
    X_test : pd.DataFrame
    n_samples : int — número de amostras para o SHAP explainer
    """
    print(f"[{nome}] Calculando SHAP values (n_samples={n_samples})...")

    X_sample = X_test.sample(min(n_samples, len(X_test)), random_state=42)

    try:
        # TreeExplainer é mais eficiente para modelos baseados em árvores
        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(X_sample)

        # Para classificação binária, shap_values pode ser lista [neg, pos]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
        plt.title(f"SHAP Summary Plot — {nome}")
        plt.tight_layout()

        caminho = FIGURES_DIR / f"shap_summary_{nome}.png"
        plt.savefig(caminho, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Figura salva: {caminho}")

    except Exception as e:
        print(f"[{nome}] Erro ao calcular SHAP: {e}. Tentando com KernelExplainer...")
        try:
            background = shap.kmeans(X_sample, 10)
            explainer = shap.KernelExplainer(modelo.predict_proba, background)
            shap_values = explainer.shap_values(X_sample.head(100))
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            fig = plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample.head(100), show=False, max_display=20)
            plt.title(f"SHAP Summary Plot (Kernel) — {nome}")
            plt.tight_layout()

            caminho = FIGURES_DIR / f"shap_summary_{nome}.png"
            plt.savefig(caminho, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Figura salva: {caminho}")
        except Exception as e2:
            print(f"[{nome}] SHAP falhou: {e2}")


def avaliar_todos_modelos(
    modelos: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    calcular_shap: bool = True,
) -> pd.DataFrame:
    """
    Avalia todos os modelos, gera todas as visualizações e retorna tabela comparativa.

    Parâmetros
    ----------
    modelos : dict { nome: estimator }
    X_test, y_test — dados de teste
    calcular_shap : bool — se True, gera SHAP plots (mais lento)

    Retorna
    -------
    pd.DataFrame com métricas comparativas
    """
    resultados = []
    feature_names = list(X_test.columns)

    for nome, modelo in modelos.items():
        metricas = calcular_metricas(nome, modelo, X_test, y_test)
        resultados.append(metricas)
        plotar_matriz_confusao(nome, modelo, X_test, y_test)
        plotar_feature_importance(nome, modelo, feature_names)
        if calcular_shap:
            plotar_shap(nome, modelo, X_test)

    plotar_curvas_roc(modelos, X_test, y_test)
    plotar_curvas_precision_recall(modelos, X_test, y_test)

    return comparar_modelos(resultados)
