"""
Quick smoke test for the CNN training stack.
"""

import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run a smoke test training job")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/smoke_test")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--config",
        args.config,
        "--output_dir",
        args.output_dir,
        "--epochs",
        str(args.epochs),
        "--num_samples",
        str(args.num_samples),
        "--batch_size",
        str(args.batch_size),
    ]
    print("[+] Running smoke test:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print(f"[✓] Smoke test passed. Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
