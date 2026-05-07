"""Forward-pass benchmark vs torch SDPA.

Run on the remote GPU:
    python bench_flash_attn.py

Prints TFLOPS and ms across a sweep of sequence lengths.
"""

import torch
import torch.nn.functional as F
import triton

from flash_attn import flash_attn_forward


def attn_flops(Z, H, N, D):
    # 2 matmuls of shape (N, D) x (D, N) per (z, h): 2 * N * D * N for QK^T,
    # then 2 * N * N * D for P @ V.
    return 4 * Z * H * N * N * D


def bench_one(Z, H, N, D, dtype, provider, causal=False):
    q = torch.randn(Z, H, N, D, device="cuda", dtype=dtype)
    k = torch.randn(Z, H, N, D, device="cuda", dtype=dtype)
    v = torch.randn(Z, H, N, D, device="cuda", dtype=dtype)
    sm_scale = 1.0 / (D ** 0.5)

    if provider == "triton":
        fn = lambda: flash_attn_forward(q, k, v, sm_scale, causal)
    elif provider == "torch":
        fn = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=sm_scale)
    else:
        raise ValueError(provider)

    # warmup
    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    tflops = attn_flops(Z, H, N, D) * 1e-12 / (ms * 1e-3)
    return ms, tflops


def main():
    Z, H, D = 2, 16, 64
    dtype = torch.float16

    print(f"Forward-only benchmark | Z={Z} H={H} D={D} dtype={dtype}")

    print(f"\n--- Non-causal ---")
    print(f"{'N':>8} | {'triton ms':>10} {'triton TF':>10} | {'torch ms':>10} {'torch TF':>10} | speedup")
    print("-" * 80)
    for N in (512, 1024, 2048, 4096, 8192, 16384):
        try:
            t_ms, t_tf = bench_one(Z, H, N, D, dtype, "triton", causal=False)
            r_ms, r_tf = bench_one(Z, H, N, D, dtype, "torch",  causal=False)
            print(f"{N:>8} | {t_ms:>10.3f} {t_tf:>10.1f} | {r_ms:>10.3f} {r_tf:>10.1f} | {r_ms/t_ms:>5.2f}x")
        except torch.cuda.OutOfMemoryError:
            print(f"{N:>8} | OOM")
            break

    print(f"\n--- Causal ---")
    print(f"{'N':>8} | {'triton ms':>10} {'triton TF':>10} | {'torch ms':>10} {'torch TF':>10} | vs torch | vs non-causal")
    print("-" * 95)
    for N in (512, 1024, 2048, 4096, 8192, 16384):
        try:
            t_ms,  t_tf  = bench_one(Z, H, N, D, dtype, "triton", causal=True)
            r_ms,  r_tf  = bench_one(Z, H, N, D, dtype, "torch",  causal=True)
            nc_ms, nc_tf = bench_one(Z, H, N, D, dtype, "triton", causal=False)
            print(f"{N:>8} | {t_ms:>10.3f} {t_tf:>10.1f} | {r_ms:>10.3f} {r_tf:>10.1f} | {r_ms/t_ms:>7.2f}x | {nc_ms/t_ms:>12.2f}x")
        except torch.cuda.OutOfMemoryError:
            print(f"{N:>8} | OOM")
            break


if __name__ == "__main__":
    main()
