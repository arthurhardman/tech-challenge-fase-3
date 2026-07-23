.PHONY: install test run run-fast run-quick lint \
        fase3 fase3-dados fase3-finetuning fase3-eval fase3-fluxo fase3-test

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

# ---- Fase 3: assistente médico (fine-tuning + LangChain/LangGraph) ---------- #
fase3:                 ## pipeline completo da Fase 3 (modo demo, offline)
	python run_fase3.py

fase3-dados:           ## gera a base sintética anonimizada
	python scripts/gen_synthetic_data.py

fase3-finetuning:      ## fine-tuning (modo demo) + curva de perda
	python -m src.assistant.finetuning.train --mode demo

fase3-eval:            ## avalia o assistente (métricas)
	python -m src.assistant.finetuning.evaluate

fase3-fluxo:           ## demonstra o fluxo LangGraph (PAC-0001)
	python -m src.assistant.cli fluxo --paciente PAC-0001

fase3-test:            ## testes da Fase 3
	python -m pytest tests/test_finetuning.py tests/test_assistant.py -v

run:
	python run_pipeline.py

run-fast:
	python run_pipeline.py --no-grid --no-shap

run-quick:
	python run_pipeline.py --no-grid --no-shap --nrows 10000

lint:
	python -m py_compile src/load_data.py src/preprocessing.py src/modeling.py src/evaluation.py run_pipeline.py tests/test_preprocessing.py tests/test_modeling.py && echo "OK"
