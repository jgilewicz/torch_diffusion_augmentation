set dotenv-load

root := justfile_directory()

default:
    @just --list

setup:
    git clone --depth=1 https://github.com/huggingface/diffusers || true
    uv pip install -e diffusers
    uv pip install -r diffusers/examples/dreambooth/requirements.txt
    uv pip install wandb huggingface_hub torchmetrics datasets pillow

download-datasets:
    uv run scripts/download_datasets.py

prepare-sana-datasets:
    uv run scripts/prepare_sana_datasets.py

datasets: download-datasets prepare-sana-datasets

stage1:
    uv run stage_1/select_model.py

stage2:
    uv run stage_2/sana_lora_finetune.py

prepare-resnet-data:
    uv run scripts/prepare_resnet_data.py

generate-synthetic:
    uv run scripts/generate_synthetic_images.py

resnet-data: prepare-resnet-data generate-synthetic

login-wandb:
    uv run wandb login

login-hf:
    uv run hf auth login 

clean:
    rm -rf outputs

clean-all: clean
    rm -rf data diffusers
