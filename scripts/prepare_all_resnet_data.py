from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).parent


def run_script(script_name: str, description: str) -> bool:
    print(f"\n{'=' * 70}")
    print(description)
    print(f"{'=' * 70}")
    try:
        subprocess.run([sys.executable, str(SCRIPTS_DIR / script_name)], check=True)
        print(f"{description} completed")
        return True
    except subprocess.CalledProcessError as error:
        print(f"{description} failed with exit code {error.returncode}")
        return False


def main() -> None:
    print("Starting resnet_data preparation pipeline...")
    if not run_script("prepare_resnet_data.py", "Prepare original and augmented datasets"):
        sys.exit(1)
    if not run_script("generate_synthetic_images.py", "Generate synthetic images with LoRA models"):
        sys.exit(1)
    print("\nresnet_data is ready.")


if __name__ == "__main__":
    main()
