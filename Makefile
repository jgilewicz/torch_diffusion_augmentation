SHELL := /bin/bash
ROOT := $(CURDIR)
PYTHONPATH := $(ROOT)/src
export PYTHONPATH

ifneq (,$(wildcard .env))
include .env
export
endif

.PHONY: help setup download-datasets prepare-sana-datasets datasets model-selection lora-finetune resnet-train resnet-eval stage1 stage2 stage3 stage3-eval prepare-resnet-data generate-synthetic resnet-data login-wandb login-hf clean clean-all

help:
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "%-24s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Install local Diffusers example dependencies.
	git clone --depth=1 https://github.com/huggingface/diffusers || true
	uv pip install -e diffusers
	uv pip install -r diffusers/examples/dreambooth/requirements.txt
	uv pip install -e .

download-datasets: ## Download source datasets into data/.
	uv run python scripts/download_datasets.py

prepare-sana-datasets: ## Build DreamBooth-style SANA datasets.
	uv run python scripts/prepare_sana_datasets.py

datasets: download-datasets prepare-sana-datasets ## Download and prepare SANA datasets.

model-selection: ## Run model selection.
	uv run python scripts/select_model.py

lora-finetune: ## Fine-tune SANA LoRAs.
	uv run python scripts/sana_lora_finetune.py

resnet-train: ## Train ResNet classifiers.
	uv run python scripts/train_resnet.py

resnet-eval: ## Evaluate ResNet classifiers.
	uv run python scripts/evaluate_resnet.py

stage1: model-selection
stage2: lora-finetune
stage3: resnet-train
stage3-eval: resnet-eval

prepare-resnet-data: ## Prepare original and augmented ResNet folders.
	uv run python scripts/prepare_resnet_data.py

generate-synthetic: ## Generate synthetic ResNet images from LoRAs.
	uv run python scripts/generate_synthetic_images.py

resnet-data: prepare-resnet-data generate-synthetic ## Prepare all ResNet data variants.

login-wandb: ## Log in to Weights & Biases.
	uv run wandb login

login-hf: ## Log in to Hugging Face.
	uv run hf auth login

clean: ## Remove generated outputs.
	rm -rf outputs

clean-all: clean ## Remove generated data and local Diffusers checkout.
	rm -rf data resnet_data diffusers
