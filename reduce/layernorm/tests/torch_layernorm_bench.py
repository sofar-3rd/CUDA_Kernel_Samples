"""Minimal PyTorch LayerNorm benchmark for ncu profiling.

Usage:
    python torch_layernorm_bench.py --shape M N [--warmup 100] [--iters 20]

Only runs torch.nn.functional.layer_norm so that ncu sees a steady stream of
the ATen LayerNorm kernels (no extension build, no correctness suite).
"""

import argparse

import torch
import torch.nn.functional as F

EPSILON = 1e-5


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", nargs=2, type=int, metavar=("M", "N"), required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    torch.manual_seed(42)
    M, N = args.shape
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    gamma = torch.randn(N, device="cuda", dtype=torch.float32)
    beta = torch.randn(N, device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        for _ in range(args.warmup):
            y = F.layer_norm(x, (N,), gamma, beta, eps=EPSILON)
        torch.cuda.synchronize()
        for _ in range(args.iters):
            y = F.layer_norm(x, (N,), gamma, beta, eps=EPSILON)
        torch.cuda.synchronize()
    print(f"[DONE] torch={torch.__version__} M={M} N={N} out={tuple(y.shape)}", flush=True)


if __name__ == "__main__":
    main()
