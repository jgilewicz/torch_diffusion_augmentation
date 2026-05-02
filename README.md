# PNW

This repository runs an image augmentation experiment based on SANA LoRA fine-tuning and ResNet evaluation.

The pipeline:

1. Download CIFAR-10, EuroSAT RGB, and Beans samples.
2. Convert them into SANA DreamBooth-style datasets.
3. Fine-tune one SANA LoRA per dataset.
4. Build ResNet training folders for original, classical augmentation, and generated images.
5. Generate synthetic images with the trained LoRAs.
6. Train and evaluate ResNet classifiers across the three data variants.

## Repository Layout

```text
configs/                         Hydra configuration files
scripts/                         CLI entry points
src/pnw/model_selection/         model selection workflow
src/pnw/lora_finetune/           SANA LoRA fine-tuning workflow
src/pnw/resnet_training/         ResNet training and evaluation workflow
src/pnw/data_prep.py             dataset preparation helpers
src/pnw/synthetic.py             synthetic image generation
src/pnw/resnet.py                shared ResNet transforms and split helpers
```

Generated folders are ignored by git:

```text
data/
resnet_data/
outputs/
diffusers/
wandb/
```

## Requirements

- Python 3.11 or newer
- `uv`
- CUDA GPU for SANA fine-tuning and generation
- Hugging Face account with access to the used models
- Weights & Biases account, unless using offline logging

Install project dependencies and the local Diffusers checkout:

```bash
make setup
```

The setup target clones `huggingface/diffusers`, installs the local checkout, installs DreamBooth example requirements, and installs this package in editable mode.

## Environment

Create `.env` from `.env.example` and fill only secrets or account-specific values:

```bash
WANDB_API_KEY=...
WANDB_PROJECT=PNW
HF_USERNAME=...
HF_ORG=
HF_TOKEN=...
```

SANA experiment parameters are stored in YAML, not `.env`.

Log in manually if preferred:

```bash
make login-wandb
make login-hf
```

For offline W&B logging:

```bash
WANDB_MODE=offline make resnet-train
```

## Configuration

Main configs:

```text
configs/download_datasets.yaml
configs/prepare_sana_datasets.yaml
configs/lora_finetune.yaml
configs/prepare_resnet_data.yaml
configs/generate_synthetic_images.yaml
configs/resnet_train.yaml
configs/resnet_eval.yaml
configs/model_selection.yaml
```

SANA LoRA defaults are in `configs/lora_finetune.yaml`:

```yaml
model_name: Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers
resolution: 512
max_train_steps: 300
lora_rank: 2
lora_alpha: 2
learning_rate: 1e-4
gradient_accumulation_steps: 1
```

The same base SANA model is used for synthetic image generation in `configs/generate_synthetic_images.yaml`.

Hydra overrides can be passed after the script command:

```bash
uv run python scripts/sana_lora_finetune.py max_train_steps=500 lora_rank=4
uv run python scripts/train_resnet.py epochs=50 batch_size=64
uv run python scripts/evaluate_resnet.py num_folds=10
```

## Reproduction

Run the full experiment in this order.

1. Install dependencies:

```bash
make setup
```

2. Download source datasets:

```bash
make download-datasets
```

This writes class folders under `data/cifar10`, `data/eurosat_rgb`, and `data/beans`.

3. Prepare SANA training datasets:

```bash
make prepare-sana-datasets
```

This writes:

```text
data/sana_cifar10/train/
data/sana_eurosat_rgb/train/
data/sana_beans/train/
```

Each folder contains images and `metadata.jsonl`.

4. Fine-tune SANA LoRAs:

```bash
make lora-finetune
```

Outputs are written to:

```text
outputs/lora_finetune/SANA-LoRA-CIFAR10/
outputs/lora_finetune/SANA-LoRA-EuroSAT-RGB/
outputs/lora_finetune/SANA-LoRA-Beans/
```

The workflow also pushes LoRAs to Hugging Face Hub. Set `HF_USERNAME` or `HF_ORG` in `.env` before running.

5. Prepare ResNet data folders:

```bash
make prepare-resnet-data
```

This writes:

```text
resnet_data/<dataset>/original/
resnet_data/<dataset>/augmented/
resnet_data/<dataset>/generated/
```

By default, `generated/` contains the original dataset images first. The synthetic images are added to the same class folders in the next step, so this variant is `original + generated`.

By default, `augmented/` also contains the original dataset images first. It then adds static classical augmentations named `augmented_*.png`, so this variant is `original + augmented`.

6. Generate synthetic images:

```bash
make generate-synthetic
```

Generated images are saved into `resnet_data/<dataset>/generated/<class>/`.

7. Train ResNet classifiers:

```bash
make resnet-train
```

Model checkpoints are written to:

```text
outputs/resnet_training/<dataset>_<variant>.pt
```

8. Evaluate ResNet classifiers:

```bash
make resnet-eval
```

Evaluation results are written to:

```text
outputs/resnet_evaluation/evaluation_results.json
```

## Useful Shortcuts

Prepare both downloaded datasets and SANA datasets:

```bash
make datasets
```

Prepare all ResNet data variants:

```bash
make resnet-data
```

Run model selection:

```bash
make model-selection
```

Clean generated outputs:

```bash
make clean
```

Remove all generated data and the local Diffusers checkout:

```bash
make clean-all
```

## Notes

- `make lora-finetune` requires `diffusers/examples/dreambooth/train_dreambooth_lora_sana.py`, so run `make setup` first.
- `make generate-synthetic` expects LoRA IDs in `configs/generate_synthetic_images.yaml`.
- ResNet training uses a fixed stratified split with `random_seed: 42` and `train_fraction: 0.8`.
- Evaluation uses 5-fold splits over the held-out original samples by default.
