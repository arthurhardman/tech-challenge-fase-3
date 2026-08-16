.PHONY: install install-fase3-local test run run-fast run-quick lint \
        fase3 fase3-local fase3-lora-dry fase3-lora-real fase3-dados-sinteticos \
        fase3-eval fase3-fluxo fase3-test

install:
	pip install -r requirements.txt

install-fase3-local:
	pip install -r requirements-fase3-local.txt

test:
	python -m pytest tests/ -v

# ---- Fase 3: assistente médico (fine-tuning + LangChain/LangGraph) ---------- #
fase3:                 ## pipeline completo com treino local real em CPU
	python run_fase3.py --mode local

fase3-local:           ## alias explícito do pipeline acima
	python run_fase3.py --mode local

fase3-lora-dry:        ## valida dataset/config do LoRA sem baixar o modelo
	python run_fase3.py --mode lora-dry-run

fase3-lora-real:       ## fine-tuning LoRA real (GPU + requirements-finetuning.txt)
	python run_fase3.py --mode lora-real

fase3-dados-sinteticos: ## gera a base sintética antiga somente quando desejado
	python scripts/gen_synthetic_data.py

fase3-eval:
	python -m src.assistant.finetuning.evaluate

fase3-fluxo:
	python -m src.assistant.cli pacientes

fase3-test:
	python -m pytest tests/test_finetuning.py tests/test_assistant.py tests/test_fase3_improvements.py -v

run:
	python run_pipeline.py

run-fast:
	python run_pipeline.py --no-grid --no-shap

run-quick:
	python run_pipeline.py --no-grid --no-shap --nrows 10000

lint:
	python -m compileall -q src tests scripts run_pipeline.py run_fase3.py && echo "OK"
