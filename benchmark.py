import os
import json
import torch
import torch.nn.functional as F
import triton
import matplotlib.pyplot as plt
import numpy as np

from flash_attn import flash_attn_forward

# A100 HBM bandwidth peak
A100_HBM_BW_GBs = 2039

def attn_flops(Z, H, N, D):
    # Q @ K ^ T -> (N, D) x (D, N) -> 2 * N * D * N FLOPs
    # P @ V -> (N, N) × (N, D) → 2 * N * N * D
    # Per (Z x H) -> 4 * N * N * D FLOPs
    return 4 * Z * H * N * N * D

def attn_bytes(Z, H, N, D, dtype):
    # Convert to bytes
    elem = torch.finfo(dtype).bits // 8
    # 3 reads (Q, K, V) + 1 write (O)
    # Each matrix (Z * H * N * D)
    # Additionally LogSumExp stored -> Z * H * N * 4 (fp32)
    return (4 * Z * H * N * D) * elem + Z * H * N * 4

def bench_memory(fn):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6

def run_sweep():
    results = []
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    head_dims = [64, 128]
    dtype = torch.float16

    for D in head_dims:
        Z, H = 2, 16 if D == 64 else 8
        for causal in [False, True]:
            for N in seq_lengths:
                z = Z if N <= 8192 else 1
                h = H if N <= 16384 else H // 2

                try:
                    q = torch.randn(z, h, N, D, device='cuda', dtype=dtype)
                    k = torch.randn(z, h, N, D, device='cuda', dtype=dtype)
                    v = torch.randn(z, h, N, D, device='cuda', dtype=dtype)
                    sm_scale = 1.0 / (D ** 0.5)

                    triton_fn = lambda: flash_attn_forward(q, k, v, sm_scale, causal)
                    torch_fn = lambda: F.scaled_dot_product_attention(q, k, v, scale=sm_scale, is_causal=causal)

                    for _ in range(5):
                        triton_fn()
                        torch_fn()
                    torch.cuda.synchronize()

                    triton_ms = triton.testing.do_bench(triton_fn, warmup=25, rep=100)
                    torch_ms = triton.testing.do_bench(torch_fn, warmup=25, rep=100)
                    triton_mem = bench_memory(triton_fn)
                    torch_mem = bench_memory(torch_fn)

                    total_bytes = attn_bytes(z, h, N, D, dtype)
                    triton_bw = total_bytes / (triton_ms * 1e-3) / 1e9
                    torch_bw = total_bytes / (torch_ms * 1e-3) / 1e9

                    flops = attn_flops(z, h, N, D)
                    triton_tf = flops * 1e-12 / (triton_ms * 1e-3)
                    torch_tf = flops * 1e-12 / (torch_ms * 1e-3)

                    result = {
                        'D': D, 'N': N, 'Z': z, 'H': h,
                        'causal': causal,
                        'triton_ms': triton_ms, 'torch_ms': torch_ms,
                        'triton_mem_mb': triton_mem, 'torch_mem_mb': torch_mem,
                        'triton_bw_gbs': triton_bw, 'torch_bw_gbs': torch_bw,
                        'triton_tflops': triton_tf, 'torch_tflops': torch_tf,
                        'speedup': torch_ms / triton_ms,
                    }
                    results.append(result)

                    tag = "causal" if causal else "non-causal"
                    print(f"D={D:3d} N={N:6d} {tag:10s} | "
                        f"triton: {triton_ms:.3f}ms {triton_bw:.0f}GB/s {triton_mem:.0f}MB | "
                        f"torch: {torch_ms:.3f}ms {torch_bw:.0f}GB/s {torch_mem:.0f}MB | "
                        f"{torch_ms/triton_ms:.2f}x")

                except torch.cuda.OutOfMemoryError:
                    print(f"D={D:3d} N={N:6d} | OOM")
                    break

    return results

def plot_results(results):
    os.makedirs('plots', exist_ok=True)
    head_dims = sorted(set(r['D'] for r in results))

    for D in head_dims:
        data = [r for r in results if r['D'] == D]

        nc = [r for r in data if not r['causal']]
        ca = [r for r in data if r['causal']]
        nc_N = [r['N'] for r in nc]
        ca_N = [r['N'] for r in ca]

        # 1. Latency vs N
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(nc_N, [r['triton_ms'] for r in nc], 'o-',  label='Triton non-causal')
        ax.plot(nc_N, [r['torch_ms']  for r in nc], 's--', label='Torch non-causal')
        ax.plot(ca_N, [r['triton_ms'] for r in ca], '^-',  label='Triton causal')
        ax.plot(ca_N, [r['torch_ms']  for r in ca], 'd--', label='Torch causal')
        ax.set_xlabel('Sequence Length (N)')
        ax.set_ylabel('Latency (ms)')
        ax.set_title(f'Forward Latency — D={D}')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(f'plots/latency_D{D}.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Peak memory vs N
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(nc_N, [r['triton_mem_mb'] for r in nc], 'o-',  label='Triton non-causal')
        ax.plot(nc_N, [r['torch_mem_mb']  for r in nc], 's--', label='Torch non-causal')
        ax.plot(ca_N, [r['triton_mem_mb'] for r in ca], '^-',  label='Triton causal')
        ax.plot(ca_N, [r['torch_mem_mb']  for r in ca], 'd--', label='Torch causal')
        ax.set_xlabel('Sequence Length (N)')
        ax.set_ylabel('Peak Memory (MB)')
        ax.set_title(f'Peak GPU Memory — D={D}')
        ax.set_xscale('log', base=2)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(f'plots/memory_D{D}.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 3. HBM bandwidth vs N
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(nc_N, [r['triton_bw_gbs'] for r in nc], 'o-',  label='Triton non-causal')
        ax.plot(nc_N, [r['torch_bw_gbs']  for r in nc], 's--', label='Torch non-causal')
        ax.plot(ca_N, [r['triton_bw_gbs'] for r in ca], '^-',  label='Triton causal')
        ax.plot(ca_N, [r['torch_bw_gbs']  for r in ca], 'd--', label='Torch causal')
        ax.axhline(y=A100_HBM_BW_GBs, color='r', linestyle=':', alpha=0.7,
                    label=f'A100 peak ({A100_HBM_BW_GBs} GB/s)')
        ax.set_xlabel('Sequence Length (N)')
        ax.set_ylabel('HBM Bandwidth (GB/s)')
        ax.set_title(f'HBM Bandwidth Utilization — D={D}')
        ax.set_xscale('log', base=2)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(f'plots/bandwidth_D{D}.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 4. Scaling (log-log with slope)
        fig, ax = plt.subplots(figsize=(10, 6))
        log_nc_N = np.log2(nc_N)
        triton_slope = np.polyfit(log_nc_N, np.log2([r['triton_ms'] for r in nc]), 1)[0]
        torch_slope = np.polyfit(log_nc_N, np.log2([r['torch_ms'] for r in nc]), 1)[0]
        ax.plot(nc_N, [r['triton_ms'] for r in nc], 'o-',
                label=f'Triton non-causal (slope={triton_slope:.2f})')
        ax.plot(nc_N, [r['torch_ms'] for r in nc], 's--',
                label=f'Torch non-causal (slope={torch_slope:.2f})')
        if ca:
            log_ca_N = np.log2(ca_N)
            triton_ca_slope = np.polyfit(log_ca_N, np.log2([r['triton_ms'] for r in ca]), 1)[0]
            ax.plot(ca_N, [r['triton_ms'] for r in ca], '^-',
                    label=f'Triton causal (slope={triton_ca_slope:.2f})')
        ax.set_xlabel('Sequence Length (N)')
        ax.set_ylabel('Latency (ms)')
        ax.set_title(f'Scaling Behavior (log-log) — D={D}\nO(N²) → slope ≈ 2.0')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(f'plots/scaling_D{D}.png', dpi=150, bbox_inches='tight')
        plt.close()

def main():
    print("FlashAttention Comprehensive Benchmark")
    print("=" * 80)

    results = run_sweep()

    os.makedirs('plots', exist_ok=True)
    with open('plots/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    plot_results(results)
    print(f"\nPlots saved to plots/")
    print(f"Raw data saved to plots/results.json")


if __name__ == '__main__':
    main()