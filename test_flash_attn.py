"""Correctness tests for the stage-1 forward kernel.

Run on the remote GPU:
    pytest -xvs test_flash_attn.py
"""

import pytest
import torch
import torch.nn.functional as F

from flash_attn import flash_attn_forward


def _reference(q, k, v, sm_scale):
    # Manual reference in fp32 for a stable ground truth.
    q32, k32, v32 = q.float(), k.float(), v.float()
    scores = (q32 @ k32.transpose(-2, -1)) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    return (probs @ v32).to(q.dtype)

def _reference_causal(q, k, v, sm_scale):
    q32, k32, v32 = q.float(), k.float(), v.float()
    scores = (q32 @ k32.transpose(-2, -1)) * sm_scale
    N = q.shape[-2]
    # Lower traingular matrix of 1's
    mask = torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool))
    # Fill masked positions with -inf
    scores = scores.masked_fill(~mask, float('-inf'))
    probs = torch.softmax(scores, dim=-1)
    return (probs @ v32).to(q.dtype)


@pytest.mark.parametrize("Z,H,N,D", [
    (1, 2, 128, 64),
    (2, 4, 256, 64),
    (1, 8, 512, 64),
    (2, 2, 1024, 128),
    (1, 4, 2048, 128),
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_forward_matches_reference(Z, H, N, D, dtype):
    torch.manual_seed(0)
    q = torch.randn(Z, H, N, D, device="cuda", dtype=dtype)
    k = torch.randn(Z, H, N, D, device="cuda", dtype=dtype)
    v = torch.randn(Z, H, N, D, device="cuda", dtype=dtype)
    sm_scale = 1.0 / (D ** 0.5)

    out, lse = flash_attn_forward(q, k, v, sm_scale)
    ref = _reference(q, k, v, sm_scale)

    # fp16/bf16 attention is noisy, especially at long N — match the tutorial's tolerance.
    atol = 1e-2 if dtype is torch.float16 else 2e-2
    torch.testing.assert_close(out, ref, atol=atol, rtol=0)

    # Sanity-check LSE shape (we'll exercise its values once the backward exists).
    assert lse.shape == (Z, H, N)
    assert lse.dtype is torch.float32

@pytest.mark.parametrize("Z,H,N,D", [
      (1, 2, 128, 64),
      (2, 4, 256, 64),
      (1, 8, 512, 64),
      (2, 2, 1024, 128),
      (1, 4, 2048, 128),
  ])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_causal_matches_reference(Z, H, N, D, dtype):
    torch.manual_seed(0)
    q = torch.randn(Z, H, N, D, device="cuda", dtype=dtype)
    k = torch.randn(Z, H, N, D, device="cuda", dtype=dtype)
    v = torch.randn(Z, H, N, D, device="cuda", dtype=dtype)
    sm_scale = 1.0 / (D ** 0.5)

    out, lse = flash_attn_forward(q, k, v, sm_scale, causal=True)
    ref = _reference_causal(q, k, v, sm_scale)

    atol = 1e-2 if dtype is torch.float16 else 2e-2
    torch.testing.assert_close(out, ref, atol=atol, rtol=0)
    assert lse.shape == (Z, H, N)
    assert lse.dtype is torch.float32

def test_causal_matches_torch_sdpa():
    torch.manual_seed(0)
    Z, H, N, D = 2, 4, 512, 64
    q = torch.randn(Z, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(Z, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(Z, H, N, D, device="cuda", dtype=torch.float16)
    sm_scale = 1.0 / (D ** 0.5)

    out, _ = flash_attn_forward(q, k, v, sm_scale, causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=sm_scale)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=0)

def test_matches_torch_sdpa():
    """Cross-check against torch.nn.functional.scaled_dot_product_attention."""
    torch.manual_seed(0)
    Z, H, N, D = 2, 4, 512, 64
    q = torch.randn(Z, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(Z, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(Z, H, N, D, device="cuda", dtype=torch.float16)
    sm_scale = 1.0 / (D ** 0.5)

    out, _ = flash_attn_forward(q, k, v, sm_scale)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=sm_scale)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=0)
