"""Measure CompactEESNN training and inference throughput on synthetic [B, 18, 1024] windows.

  python scripts/benchmark_snn.py                       # n_hidden 1 and 2, batch 64
  python scripts/benchmark_snn.py --batch-sizes 64 256 --windows 3500000

Reports windows/second, peak GPU memory, and the time for one pass over --windows
(default ~3.5M: all usable CHB-MIT hours at a 1 s stride).
"""
from __future__ import annotations

import argparse
import time

import torch

import _common  # noqa: F401
from src.models.snn import CompactEESNN, load_snn_config

C, T = 18, 1024


def timed(step, n_iters: int, device: torch.device) -> float:
    for _ in range(3):  # warm-up: cuDNN autotuning, allocator
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_iters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-hidden", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[64])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--windows", type=float, default=3.5e6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    cfg = load_snn_config()["model"]
    print(f"device: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}")
    print(f"{'hidden':>6} {'batch':>5} {'params':>7} {'train win/s':>11} {'infer win/s':>11} "
          f"{'peak MB':>7} {'train pass':>10} {'infer pass':>10}")

    for n_hidden in args.n_hidden:
        for bs in args.batch_sizes:
            torch.manual_seed(0)
            model = CompactEESNN({**cfg, "n_hidden": n_hidden}).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = torch.nn.BCEWithLogitsLoss()
            x = torch.randn(bs, C, T, device=device)
            y = (torch.rand(bs, device=device) < 0.2).float()

            def train_step():
                opt.zero_grad(set_to_none=True)
                loss_fn(model(x), y).backward()
                opt.step()

            @torch.no_grad()
            def infer_step():
                model(x)

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model.train()
            t_train = timed(train_step, args.iters, device)
            model.eval()
            t_infer = timed(infer_step, args.iters, device)
            peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else float("nan")

            train_rate, infer_rate = bs / t_train, bs / t_infer
            print(f"{n_hidden:>6} {bs:>5} {model.count_params():>7} {train_rate:>11.0f} {infer_rate:>11.0f} "
                  f"{peak:>7.0f} {args.windows / train_rate / 60:>8.1f} m {args.windows / infer_rate / 60:>8.1f} m")


if __name__ == "__main__":
    main()
