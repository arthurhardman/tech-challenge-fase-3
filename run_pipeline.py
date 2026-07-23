"""
run_pipeline.py
---------------
Executa o pipeline completo de pré-processamento, treinamento e avaliação.

Uso:
    python run_pipeline.py                  # pipeline completo com GridSearchCV
    python run_pipeline.py --no-grid        # treino rápido sem busca de hiperparâmetros
    python run_pipeline.py --no-shap        # pula SHAP (mais rápido)
    python run_pipeline.py --data caminho/arquivo.csv

Paralelismo no GridSearchCV:
    export SRAG_GRID_N_JOBS=4 && python run_pipeline.py
"""

import argparse
import sys
from pathlib import Path

# Garante que src/ está no path mesmo rodando da raiz do projeto
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.tabular.load_data import carregar_dataset
from src.tabular.preprocessing import executar_pipeline_preprocessamento
from src.tabular.modeling import treinar_todos_modelos
from src.tabular.evaluation import avaliar_todos_modelos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline SRAG — classificação de desfecho clínico")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Caminho para o CSV bruto (padrão: data/raw/INFLUD24-26-06-2025.csv)",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Desativa GridSearchCV (treino rápido com hiperparâmetros padrão)",
    )
    parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Pula geração de SHAP plots (reduz tempo de avaliação)",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Limita número de linhas carregadas (útil para teste rápido)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("PIPELINE SRAG — Classificação de Desfecho Clínico")
    print("=" * 60)

    # 1. Carregamento
    print("\n[1/3] Carregando dataset...")
    kwargs = {"nrows": args.nrows} if args.nrows else {}
    if args.data:
        df = carregar_dataset(caminho=args.data, **kwargs)
    else:
        df = carregar_dataset(**kwargs)

    # 2. Pré-processamento
    print("\n[2/3] Pré-processando...")
    artefatos = executar_pipeline_preprocessamento(df, salvar=True)

    X_train = artefatos["X_train"]
    X_test  = artefatos["X_test"]
    y_train = artefatos["y_train"]
    y_test  = artefatos["y_test"]

    # 3. Treinamento
    print("\n[3/3] Treinando modelos...")
    usar_grid = not args.no_grid
    modelos = treinar_todos_modelos(X_train, y_train, usar_grid_search=usar_grid, salvar=True)

    # 4. Avaliação
    print("\n[4/4] Avaliando modelos...")
    calcular_shap = not args.no_shap
    df_metricas = avaliar_todos_modelos(modelos, X_test, y_test, calcular_shap=calcular_shap)

    print("\n" + "=" * 60)
    print("Pipeline concluído.")
    print(f"Figuras salvas em: results/figures/")
    print(f"Modelos salvos em: results/models/")
    print("=" * 60)
    print("\nMétricas finais (teste):")
    print(df_metricas.to_string())


if __name__ == "__main__":
    main()
