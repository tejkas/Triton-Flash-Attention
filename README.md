# Triton FlashAttention

A from-scratch, staged implementation of FlashAttention in [Triton](https://triton-lang.org/),
built as a learning exercise from the
[official `06-fused-attention` tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).

The goal is to understand the algorithm by building it up one piece at a time
(non-causal forward → causal → backward → tuning), then use the finished kernel
as a baseline for comparing other attention variants (sliding window, gated,
linear, compressed, multi-latent) on a Lambda Labs A100.

See [`ROADMAP.md`](./ROADMAP.md) for the full plan and current status.

## Repository layout

```
flash_attn.py          # Triton kernel + Python wrapper (stage 1: non-causal forward)
test_flash_attn.py     # Correctness tests vs torch SDPA and an fp32 reference
bench_flash_attn.py    # Forward TFLOPS / latency vs torch SDPA
ROADMAP.md             # Staged plan and parking lot
requirements.txt
```

## Status

- **Stage 1 — Non-causal forward**: complete. fp16/bf16, head dim ∈ {16, 32, 64, 128, 256}, requires `N` to be a multiple of the block size.
- **Stage 2+**: see `ROADMAP.md`.

## Algorithm in one paragraph

FlashAttention avoids materializing the full N×N attention matrix by walking
K/V in tiles and maintaining three running statistics per query block: the
softmax max `m_i`, the denominator `l_i`, and the output accumulator `acc`.
When a new tile arrives, both the old denominator and the old output are
rescaled by `exp(m_old - m_new)` so that everything is expressed relative to
the latest max. At the end, `acc / l_i` is the exact softmax-attention output.
The five-line update is annotated in `flash_attn.py`. We track `m_i`/`l_i` in
log2 space because `exp2` is faster than `exp` on NVIDIA GPUs.

## Running it (remote GPU)

This project develops locally on macOS but runs on a remote NVIDIA GPU
(targeting an A100 on Lambda Labs). Triton kernels cannot run on Mac.

```sh
# on the remote box
pip install -r requirements.txt

# correctness
pytest -xvs test_flash_attn.py

# performance
python bench_flash_attn.py
```

The benchmark prints a table of forward latency / TFLOPS / speedup vs
`torch.nn.functional.scaled_dot_product_attention` across a sweep of sequence
lengths.

## API

```python
from flash_attn import flash_attn_forward

# q, k, v: (Z, H, N, D) on CUDA, fp16 or bf16
out, lse = flash_attn_forward(q, k, v, sm_scale=None)
# out: (Z, H, N, D)  — attention output, same dtype as q
# lse: (Z, H, N)     — logsumexp in log2 space, kept for the future backward pass
```

`sm_scale` defaults to `1/sqrt(D)`.

## References

- Dao et al., [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [Triton tutorial 06](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- Milakov & Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
