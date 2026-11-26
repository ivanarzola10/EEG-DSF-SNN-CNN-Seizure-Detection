"""Calibrate on validation first; then evaluate frozen rules on held-out patients."""
import argparse
from pathlib import Path

from src.evaluation.pipeline import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["calibrate", "test"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/prepared_proposal"))
    parser.add_argument("--experiment-config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--mask-manifest", type=Path, help="Defaults to evaluation_masks.json beside the checkpoint")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if args.batch_size <= 0 or min(args.workers, args.threads, args.log_every) < 0:
        parser.error("Batch size must be positive and worker/thread/log settings nonnegative")
    if args.mask_manifest is None:
        args.mask_manifest = args.checkpoint.parent / "evaluation_masks.json"
    run(args)


if __name__ == "__main__":
    main()
