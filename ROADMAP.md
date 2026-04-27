# Roadmap

A staged build of FlashAttention in Triton, ending in a benchmark study against
other attention variants on a Lambda Labs A100. Each stage should land
correctness tests before the next one starts.

## Stage 1 — Non-causal forward ✅

- [x] `_attn_fwd_kernel` with online-softmax loop in log2 space
- [x] `flash_attn_forward(q, k, v)` wrapper
- [x] Correctness tests vs fp32 reference and `F.scaled_dot_product_attention`
- [x] Forward-only benchmark vs torch SDPA across `N`

**Known limitations** (intentional for now):
- `N` must be a multiple of `BLOCK_M`/`BLOCK_N` (no ragged-tail masking)
- No autotune; block sizes are hardcoded heuristics
- fp16/bf16 only

## Stage 2 — Causal masking

- [ ] Add `causal: bool` to the wrapper
- [ ] Two-stage inner loop: off-band tiles unmasked, diagonal tile masked,
      above-diagonal tiles skipped entirely
- [ ] Extend tests to cover `causal=True` and ragged `N` (drop the multiple-of-BLOCK assertion)
- [ ] Re-run benchmark; expect ~2× speedup at long `N` from skipped tiles

## Stage 3 — Backward pass

- [ ] `_attn_bwd_preprocess` — compute `delta = rowsum(O * dO)`
- [ ] `_attn_bwd_dkdv` — iterate Q tiles, accumulate dK/dV
- [ ] `_attn_bwd_dq` — iterate K/V tiles, accumulate dQ
- [ ] Wire into `torch.autograd.Function` so `flash_attn(q,k,v)` is end-to-end differentiable
- [ ] Gradcheck against autograd on the fp32 reference
- [ ] Benchmark backward TFLOPS vs torch SDPA backward

## Stage 4 — Tuning

- [ ] `@triton.autotune` over `(BLOCK_M, BLOCK_N, num_warps, num_stages)`
- [ ] Sweep `D ∈ {64, 128}` separately — different sweet spots
- [ ] Optional: `tl.dot` TMA path for Hopper (skip on A100)

## Stage 5 — Compare with other attention variants

For each variant, write a minimal Triton (or torch) implementation, validate
correctness against the reference formulation, and add it to the benchmark
sweep. Goal is intuition for the speed/quality tradeoffs, not a paper-grade
study.

| Variant | One-line idea | Asymptotic |
|---|---|---|
| **Sliding Window** (Longformer / Mistral) | Each query attends only to a fixed window of nearby keys | O(N·W) |
| **Gated Attention** (GAU / GLA) | Multiplicative gate on attention output (or on the recurrent linear-attention state) | O(N²) or O(N) |
| **Linear Attention** (Performer / RetNet) | Replace softmax with kernel feature map φ; compute φ(Q) (φ(K)ᵀV) | O(N·D²) |
| **Compressed Attention** | Compress past KV (e.g., low-rank or learned downsample) before attending | O(N·M), M << N |
| **Multi-Latent Attention** (DeepSeek-V2/V3) | Project K,V into a shared low-rank latent cached at inference | O(N²) compute, O(N·R) cache |

Per-variant checklist:
- [ ] Sliding window — start here, easiest mod to existing kernel (just clamp the K/V loop range)
- [ ] Linear attention — different algorithm; good contrast vs FA
- [ ] Gated attention — pick one concrete form (GAU is simplest)
- [ ] Multi-latent attention — torch reference is fine; mainly a memory story
- [ ] Compressed attention — last; pick one method to avoid scope creep

## Stage 6 — A100 benchmark study

- [ ] Lock down GPU (Lambda Labs A100 40GB or 80GB — record which)
- [ ] Sweep: `N ∈ {512, 1k, 2k, 4k, 8k, 16k, 32k}`, `D ∈ {64, 128}`, fp16 + bf16
- [ ] Measure forward and backward separately
- [ ] Capture peak memory alongside latency
- [ ] Plot with matplotlib: TFLOPS vs N (forward, backward), peak memory vs N
- [ ] Save plots to `plots/` and raw numbers to `results.json`

## Open questions / parking lot

- Do we need varlen (cu_seqlens) support? Not yet — assume packed dense.
- Multi-query / grouped-query attention (MQA/GQA)? Out of scope unless we end up benchmarking Mistral-style models.
- FP8? Out of scope on A100 (Hopper-only).
