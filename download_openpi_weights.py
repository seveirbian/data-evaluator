#!/usr/bin/env python3
"""Download openpi model weights from GCS."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download openpi model weights")
    parser.add_argument(
        "model",
        nargs="?",
        default="pi05_droid",
        choices=["pi05_droid", "pi05_base", "pi0_droid", "pi0_base"],
        help="Model to download (default: pi05_droid)",
    )
    args = parser.parse_args()

    # Check if openpi is installed
    try:
        from openpi.shared.download import maybe_download
    except ImportError:
        print("Error: openpi is not installed.")
        print("Install with: pip install 'data-evaluator[openpi]'")
        sys.exit(1)

    # Download checkpoint
    gcs_path = f"gs://openpi-assets/checkpoints/{args.model}"
    print(f"Downloading {args.model} from {gcs_path}...")

    try:
        local_path = maybe_download(gcs_path)
        print(f"\n✓ Downloaded to: {local_path}")
        print(f"\nUse with:")
        if "jax" not in args.model:
            print(f"  python main.py --openpi-jax {local_path}")
        else:
            print(f"  python main.py --openpi-jax {local_path}")
    except Exception as e:
        print(f"Error downloading: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
