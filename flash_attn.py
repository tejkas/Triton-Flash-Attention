"""
FlashAttention 2.0 Triton implementation, including:
- Forward pass,
- Causal masking,
- Backwards pass

Inputs are laid out as (Z, H, N, D):
    Z = batch size
    H = number of heads
    N = sequence length (must be a multiple of BLOCK_N)
    D = head dimension (16, 32, 64, 128, or 256)

stride_z = H*N*D   ← skip a whole (H, N, D) slab
stride_h = N*D     ← skip a whole (N, D) head
stride_m = D       ← skip one row (one token)
stride_k = 1       ← move one element along D (contiguous)

The kernel walks K/V tiles once, maintaining the running softmax max (m_i),
denominator (l_i), and weighted-V accumulator (acc) — never materializing the
full N x N attention matrix. All running stats live in log2 space because
exp2 is faster than exp on NVIDIA GPUs.
"""

import torch
import triton
import triton.language as tl


# log2(e) — multiply scores by this once so we can use exp2 in the loop.
LOG2E = 1.4426950408889634


@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, M,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    H, N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL_MASK: tl.constexpr,
):
    start_m = tl.program_id(0)        # which BLOCK_M of queries
    off_hz = tl.program_id(1)         # flattened (batch, head)
    off_z = off_hz // H     # which batch
    off_h = off_hz % H      # which attention head

    # Skip to the (z, h) slab in each tensor.
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    # K is loaded transposed (HEAD_DIM, BLOCK_N) so that q @ k yields (BLOCK_M, BLOCK_N).
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=O + o_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # Online-softmax state. m_i and l_i are tracked in log2 space.
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Logs base 2 is easier to calculate
    qk_scale = sm_scale * LOG2E   # fold log2(e) into the scale once

    # Q stays resident in registers/SRAM for the whole loop.
    q = tl.load(Q_block_ptr)

    #  Implement causal masking. 
    #  boundary is the maximum position in K/V whose entire block can be used to compute attention
    #  i.e, block boundary is < start_m
    if CAUSAL_MASK:
        # If masking, we can completely use K/V tokens until this point
        boundary = start_m * BLOCK_M
    else:
        # Else, use full sequence N_CTX is sequence length
        boundary = N_CTX

    # Iterate K/V to process masked blocks
    for start_n in range(0, boundary, BLOCK_N):
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)

        # k is already transposed in the block pointer initialization
        # scores for this tile, fp32 accumulator
        qk = tl.dot(q, k) 

        # Online Softmax update step

        # running max
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1) * qk_scale)
        # normalized local probabilities
        p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
        # correction scaling factor
        alpha = tl.math.exp2(m_i - m_ij)

        # update running sum and output
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc) # acc += p @ v
        # set new max
        m_i = m_ij

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # Iterate K/V to process diagonal blocks (with a mix of masked and unmasked values)
    if CAUSAL_MASK:
        # Current q index
        q_idx = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        # Iterate K/V (only one block of Q)
        for start_n in range(start_m * BLOCK_M, (start_m + 1) * BLOCK_M, BLOCK_N):
            k = tl.load(K_block_ptr)
            v = tl.load(V_block_ptr)

            qk = tl.dot(q, k) 

            k_idx = start_n + tl.arange(0, BLOCK_N)
            # Causal Masking rule: Q index needs to be >= than the index of K
            causal_mask = q_idx[:, None] >= k_idx[None, :]
            qk = tl.where(causal_mask, qk, -1e6) # Set masked values to -inf to let softmax probabilities flow to 0

            # Remainder is just normal softmax update logic
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1) * qk_scale)
            p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(v.dtype), v, acc) # acc += p @ v
            m_i = m_ij

            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    # Final normalize: divide by the accumulated denominator.
    acc = acc / l_i[:, None]

    # Save logsumexp (log2-space) for the future backward pass.
    m_ptrs = M + off_hz * N_CTX + start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(m_ptrs, m_i + tl.math.log2(l_i))

    tl.store(O_block_ptr, acc.to(O.type.element_ty))

# Remember that dS = P ⊙ (dP - rowsum(<P, dP>)) where ⊙ is elementwise multiply
# Rowwise, dS_i = P_i ⊙ (dP_i - rowsum(<P_i, dP_i>))
# Preprocesses the delta_i vector (s.t delta_i = rowsum(<P_i, dP_i>) for all rows i in P)
# delta_i is just rowsum(<P_i, dP_i> = Σ_j P_ij * dP_ij
#                                    = Σ_j P_ij * (dO_i · V_j)
#                                    = dO_i · (Σ_j P_ij * V_j)
#                                    = dO_i · O_i
@triton.jit
def _attn_bwd_preprocess(
    O, dO, Delta,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_doz, stride_doh, stride_dom, stride_dok,
    H, N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    start_m = tl.program_id(0)        # which BLOCK_M of queries
    off_hz = tl.program_id(1)         # flattened (batch, head)
    off_z = off_hz // H     # which batch
    off_h = off_hz % H      # which attention head

    o_offset = off_z * stride_oz + off_h * stride_oh
    # dO starts at the same batch/head offsets as O
    do_offset = off_z * stride_doz + off_h * stride_doh

    O_block_ptr = tl.make_block_ptr(
        base=O + o_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        base=dO + do_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_dom, stride_dok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # accumulate as fp32
    o = tl.load(O_block_ptr).to(tl.float32) 
    do = tl.load(dO_block_ptr).to(tl.float32)

    delta = tl.sum(o * do, axis=1) # element wise multiply. BUT, sum across the HEAD_DIM axis.

    # Address to HBM and store
    delta_ptrs = Delta + off_hz * N_CTX + start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(delta_ptrs, delta)

# dK, dV kernel for backprop through K, V
@triton.jit
def _attn_bwd_dkdv(
      Q, K, V, sm_scale,
      dO, dK, dV,
      LSE, Delta,
      stride_qz, stride_qh, stride_qm, stride_qk,
      stride_kz, stride_kh, stride_kn, stride_kk,
      stride_vz, stride_vh, stride_vn, stride_vk,
      stride_doz, stride_doh, stride_dom, stride_dok,
      stride_dkz, stride_dkh, stride_dkn, stride_dkk,
      stride_dvz, stride_dvh, stride_dvn, stride_dvk,
      H, N_CTX,
      HEAD_DIM: tl.constexpr,
      BLOCK_M: tl.constexpr,
      BLOCK_N: tl.constexpr,
      IS_CAUSAL: tl.constexpr,
  ):
    start_n = tl.program_id(0) # which BLOCK_N of K/V
    off_hz  = tl.program_id(1) # flattened (batch, head)
    off_z   = off_hz // H # which batch
    off_h   = off_hz % H # which attention head

    # offsets
    q_offset  = off_z * stride_qz  + off_h * stride_qh
    k_offset  = off_z * stride_kz  + off_h * stride_kh
    v_offset  = off_z * stride_vz  + off_h * stride_vh
    do_offset = off_z * stride_doz + off_h * stride_doh
    dk_offset = off_z * stride_dkz + off_h * stride_dkh
    dv_offset = off_z * stride_dvz + off_h * stride_dvh

    # K, V - instantiate pointers, load from HBM, allocate mem
    K_block_ptr = tl.make_block_ptr(
          base=K + k_offset,
          shape=(HEAD_DIM, N_CTX),
          strides=(stride_kk, stride_kn),
          offsets=(0, start_n * BLOCK_N),
          block_shape=(HEAD_DIM, BLOCK_N),
          order=(0, 1),
      )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )

    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    # Q, dO - instantiate pointers, load from HBM, allocate mem
    Q_block_ptr = tl.make_block_ptr(
          base=Q + q_offset,
          shape=(N_CTX, HEAD_DIM),
          strides=(stride_qm, stride_qk),
          offsets=(0, 0),
          block_shape=(BLOCK_M, HEAD_DIM),
          order=(1, 0),
      )
    dO_block_ptr = tl.make_block_ptr(
        base=dO + do_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_dom, stride_dok),
        offsets=(0, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    lse_ptrs   = LSE   + off_hz * N_CTX + tl.arange(0, BLOCK_M)
    delta_ptrs = Delta + off_hz * N_CTX + tl.arange(0, BLOCK_M)

    for start_m in range(0, N_CTX, BLOCK_M):
        q = tl.load(Q_block_ptr)
        do = tl.load(dO_block_ptr)
        lse = tl.load(lse_ptrs)
        delta = tl.load(delta_ptrs)

        # Calculate P tile on the fly from Q, K and logsumexp, don't need to materialize full P matrix
        qk = tl.dot(q, k)
        p = tl.math.exp2(qk * (sm_scale * LOG2E) - lse[:, None])

        # dV += P^T @ dO (Chain Rule)
        dv = tl.dot(tl.trans(p).to(do.dtype), do, dv)

        # softmax backward pass - dS = P ⊙ (dP - delta) where ⊙ is elementwise multiplication
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp.to(tl.float32) - delta[:, None])

        # dK += dS^T @ Q * sm_scale
        dk += tl.dot(tl.trans(ds).to(q.dtype), q) * sm_scale

        Q_block_ptr  = tl.advance(Q_block_ptr,  (BLOCK_M, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (BLOCK_M, 0))
        lse_ptrs   += BLOCK_M
        delta_ptrs += BLOCK_M
    
    dK_block_ptr = tl.make_block_ptr(
          base=dK + dk_offset,
          shape=(N_CTX, HEAD_DIM), # no transpose needed for storing dK
          strides=(stride_dkn, stride_dkk),
          offsets=(start_n * BLOCK_N, 0),
          block_shape=(BLOCK_N, HEAD_DIM),
          order=(1, 0),
      )
    dV_block_ptr = tl.make_block_ptr(
        base=dV + dv_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_dvn, stride_dvk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    tl.store(dV_block_ptr, dv.to(v.dtype))
    tl.store(dK_block_ptr, dk.to(k.dtype))

def flash_attn_forward(q, k, v, sm_scale=None, causal=False):
    """FlashAttention forward.

    Args:
        q, k, v: (Z, H, N, D) tensors, contiguous, fp16 or bf16.
        sm_scale: softmax scale; defaults to 1/sqrt(D).

    Returns:
        o: (Z, H, N, D) attention output, same dtype as q.
        lse: (Z, H, N) logsumexp in log2 space (kept for the backward pass).
    """
    assert q.shape == k.shape == v.shape, "Q, K, V must share shape"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "tensors must be on CUDA"
    assert q.dtype in (torch.float16, torch.bfloat16), "stage 1 supports fp16/bf16 only"

    Z, H, N, D = q.shape
    assert D in {16, 32, 64, 128, 256}, f"unsupported head dim {D}"

    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    BLOCK_M = 128 if D <= 64 else 64
    BLOCK_N = 64 if D <= 64 else 32
    assert N % BLOCK_N == 0, (
        f"stage 1 requires N ({N}) to be a multiple of BLOCK_N ({BLOCK_N}); "
        "ragged tails are handled in a later stage"
    )
    assert N % BLOCK_M == 0, (
        f"stage 1 requires N ({N}) to be a multiple of BLOCK_M ({BLOCK_M})"
    )

    o = torch.empty_like(q)
    lse = torch.empty((Z, H, N), device=q.device, dtype=torch.float32) #Where logsumexp vector is stored

    # Tiling Q matrix, N/BLOCK_M
    grid = (triton.cdiv(N, BLOCK_M), Z * H, 1)
    _attn_fwd_kernel[grid](
        q, k, v, o, lse,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H, N,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CAUSAL_MASK=causal,
        num_warps=4 if D <= 64 else 8,
        num_stages=3,
    )
    return o, lse
