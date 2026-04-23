set dotenv-load

root := justfile_directory()

default:
    @just --list

setup:
    git clone --depth=1 https://github.com/huggingface/diffusers || true
    uv pip install -e diffusers
    uv pip install -r diffusers/examples/dreambooth/requirements.txt
    uv pip install wandb huggingface_hub torchmetrics

stage1:
    uv run stage_1/select_model.py

login-wandb:
    uv run wandb login

login-hf:
    uv run hf auth login 

clean:
    rm -rf outputs

clean-all: clean
    rm -rf data diffusers
