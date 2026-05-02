from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).parent


def run_command(script_path: Path, description: str) -> bool:
    print(f"\n{'=' * 70}")
    print(f"{description}")
    print(f"{'=' * 70}")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
        )
        print(f"{description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f" {description} failed with exit code {e.returncode}")
        return False


def main() -> None:

    print("Starting resnet_data preparation pipeline...")

    success = run_command(
        SCRIPTS_DIR / "prepare_resnet_data.py", "Prepare normal and augmented datasets"
    )

    if not success:
        print("\nPipeline failed at dataset preparation step")
        sys.exit(1)

    success = run_command(
        SCRIPTS_DIR / "generate_synthetic_images.py",
        "Generate synthetic images with LoRA models",
    )

    if not success:
        print(
            "\nSynthetic image generation had issues, but normal/augmented data is ready"
        )
        sys.exit(1)

    print("\n" + "=" * 70)
    print("All done! resnet_data is ready with:")
    print("   - normal/    (original images)")
    print("   - augmented/ (20 images per class with traditional augmentation)")
    print("   - diffused/  (20 images per class generated with LoRA)")
    print("=" * 70)


if __name__ == "__main__":
    main()
