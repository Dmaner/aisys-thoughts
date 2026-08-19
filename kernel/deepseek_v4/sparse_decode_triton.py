# KV cache block
# ============================================================================
# <sglang/python/sglang/kernels/ops/attention/dsv4/dequant_k_cache.py>
# v4 KV cache layout (see dsv4.index_buf_accessor._set_k_and_s_triton_kernel):
#   per-token: 448 fp8 nope + 64 bf16 rope (= 576 contiguous bytes) +
#              7 ue8m0 scales padded to 8 bytes.
#   per-page:  [token 0..P-1 nope+rope (P*576 bytes)] [token 0..P-1 scale (P*8 bytes)]
#              padded up to a multiple of 576.
# ============================================================================
# KV Cache Page
# ┌──────────────────────────────────────────────┐
# │ token 0 data：576 bytes                      │
# │ token 1 data：576 bytes                      │
# │ ...                                          │
# │ token P-1 data：576 bytes                    │
# ├──────────────────────────────────────────────┤
# │ token 0 scales：8 bytes                      │
# │ token 1 scales：8 bytes                      │
# │ ...                                          │
# │ token P-1 scales：8 bytes                    │
# ├──────────────────────────────────────────────┤
# │ page padding                                 │
# └──────────────────────────────────────────────┘

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda")

# deepseek v4 constant
DSV4_HIDDEN_DIM = 512
SM_SCALE: float = DSV4_HIDDEN_DIM**-0.5


def _bucket_total_tokens(total_token: int) -> int:
    if total_token <= 0:
        return 1
    n = 1
    while n < total_token:
        n <<= 1
    return n


# KV Cache Shape
# SWA：  [num_pages, 128, 1, 584]
# C4：   [num_pages,  64, 1, 584]
# C128： [num_pages,   2, 1, 584]
def launch_sparse_fused_gather_attention_decode(
    Q: torch.Tensor,  # [B * S, q_num_heads, head_dim]
    KV_Cache: torch.Tensor,
    Indices: torch.Tensor,  # [B * S, topk]
    BlockSize: int,
    TopkLength: torch.Tensor | None = None,
    Seq_len: int = 1,
):
    # prepare param
    q_t, q_h, _ = Q.shape
    topk = Indices.shape[1]

    kv_uint8 = KV_Cache.view(torch.uint8)
    num_blocks = kv_uint8.shape[0]
    stride_kv_block = kv_uint8.stride(0)
    kv_flat = kv_uint8.reshape(num_blocks, -1)
    # make sure q, indices is contiguous
    if Q.dtype != torch.bfloat16 or not Q.is_contiguous():
        Q = Q.to(torch.bfloat16).contiguous()
    if not Indices.is_contiguous():
        Indices = Indices.contiguous()

    output = torch.empty(
        q_t,
        q_h,
        DSV4_HIDDEN_DIM,
        dtype=torch.bfloat16,
        device=Q.device,
    )
    lse = torch.empty(q_t, q_h, dtype=torch.float32, device=Q.device)

    topk_length_tensor = TopkLength if TopkLength is not None else lse[:1, 0]

    grid = lambda meta: (triton.cdiv(q_h, meta["BLOCK_H"]), q_t)
    sparse_fused_gather_attention_kernel[grid](
        # Q
        Q,
        q_h,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        # KV
        kv_flat,
        stride_kv_block,
        num_blocks,
        # Indices
        Indices,
        Indices.stride(0),
        Indices.stride(1),
        # Output
        output,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        # LSE
        lse,
        lse.stride(0),
        lse.stride(1),
        # Topk
        topk_length_tensor,
        topk,
        # Other
        BlockSize,
        Seq_len,
        SM_SCALE,
        _bucket_total_tokens(q_t),
        HAS_TOPK_LENGTH=TopkLength is not None,
    )
    return output, lse


@triton.jit
def process_tile(
    kv_block_base,
    token_data_offset,
    token_scale_offset,
    valid,
    valid_2d,
    # Q
    q_0,
    q_1,
    q_2,
    q_3,
    q_4,
    q_5,
    q_6,
    q_7,
    # accumulators
    acc_0,
    acc_1,
    acc_2,
    acc_3,
    acc_4,
    acc_5,
    acc_6,
    acc_7,
    # softmax state
    m_i,
    l_i,
    # other param
    offest_tile,
    sm_scale,
    # constants
    TILE_SIZE: tl.constexpr,
    ROPE_OFFSET: tl.constexpr,
    LOG2E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLCOK_N: tl.constexpr,
):
    ENG_INF = float("-inf")
    # load kv scale (only for nope)
    scale_ptrs = kv_block_base + token_scale_offset
    scale_uint8_0 = tl.load(scale_ptrs, mask=valid, other=127).to(tl.uint8)
    scale_uint8_1 = tl.load(scale_ptrs + 1, mask=valid, other=127).to(tl.uint8)
    scale_uint8_2 = tl.load(scale_ptrs + 2, mask=valid, other=127).to(tl.uint8)
    scale_uint8_3 = tl.load(scale_ptrs + 3, mask=valid, other=127).to(tl.uint8)
    scale_uint8_4 = tl.load(scale_ptrs + 4, mask=valid, other=127).to(tl.uint8)
    scale_uint8_5 = tl.load(scale_ptrs + 5, mask=valid, other=127).to(tl.uint8)
    scale_uint8_6 = tl.load(scale_ptrs + 6, mask=valid, other=127).to(tl.uint8)
    # load kv block data (nope + rope)

    # tile_base [i] = page_i + token_data_slot_i
    tile_base = kv_block_base[:, None] + token_data_offset[:, None]
    # shape [BLOCK, TILE_SIZE]
    nope_uint8_0 = tl.load(tile_base + offest_tile[None, :], mask=valid_2d, other=0)
    nope_uint8_1 = tl.load(tile_base + TILE_SIZE + offest_tile[None, :], mask=valid_2d, other=0)
    nope_uint8_2 = tl.load(tile_base + 2 * TILE_SIZE + offest_tile[None, :], mask=valid_2d, other=0)
    nope_uint8_3 = tl.load(tile_base + 3 * TILE_SIZE + offest_tile[None, :], mask=valid_2d, other=0)
    nope_uint8_4 = tl.load(tile_base + 4 * TILE_SIZE + offest_tile[None, :], mask=valid_2d, other=0)
    nope_uint8_5 = tl.load(tile_base + 5 * TILE_SIZE + offest_tile[None, :], mask=valid_2d, other=0)
    nope_uint8_6 = tl.load(tile_base + 6 * TILE_SIZE + offest_tile[None, :], mask=valid_2d, other=0)

    # rope 16 = rope lo | rope hi << 8
    rope_lo = tl.load(
        tile_base + ROPE_OFFSET + offest_tile[None, :] * 2,
        mask=valid_2d,
        other=0,
    ).to(tl.uint16)
    rope_hi = tl.load(
        tile_base + ROPE_OFFSET + offest_tile[None, :] * 2 + 1, mask=valid_2d, other=0
    ).to(tl.uint16)
    # dequantize x = bf16(q) * scale
    scale_bf16_0 = tl.math.exp2(scale_uint8_0.to(tl.float32) - 127).to(tl.bfloat16)
    scale_bf16_1 = tl.math.exp2(scale_uint8_1.to(tl.float32) - 127).to(tl.bfloat16)
    scale_bf16_2 = tl.math.exp2(scale_uint8_2.to(tl.float32) - 127).to(tl.bfloat16)
    scale_bf16_3 = tl.math.exp2(scale_uint8_3.to(tl.float32) - 127).to(tl.bfloat16)
    scale_bf16_4 = tl.math.exp2(scale_uint8_4.to(tl.float32) - 127).to(tl.bfloat16)
    scale_bf16_5 = tl.math.exp2(scale_uint8_5.to(tl.float32) - 127).to(tl.bfloat16)
    scale_bf16_6 = tl.math.exp2(scale_uint8_6.to(tl.float32) - 127).to(tl.bfloat16)
    # tiled qkv
    qk = tl.zeros([BLOCK_H, BLCOK_N], dtype=tl.float32)

    nope_fp8_0 = nope_uint8_0.to(tl.float8e4nv, bitcast=True)
    kv_0 = (nope_fp8_0.to(tl.bfloat16) * scale_bf16_0[:, None]).to(tl.bfloat16)
    kv_0 = tl.where(valid_2d, kv_0, 0.0)
    qk += tl.dot(q_0, tl.trans(kv_0)).to(tl.float32)

    nope_fp8_1 = nope_uint8_1.to(tl.float8e4nv, bitcast=True)
    kv_1 = (nope_fp8_1.to(tl.bfloat16) * scale_bf16_1[:, None]).to(tl.bfloat16)
    kv_1 = tl.where(valid_2d, kv_1, 0.0)
    qk += tl.dot(q_1, tl.trans(kv_1)).to(tl.float32)

    nope_fp8_2 = nope_uint8_2.to(tl.float8e4nv, bitcast=True)
    kv_2 = (nope_fp8_2.to(tl.bfloat16) * scale_bf16_2[:, None]).to(tl.bfloat16)
    kv_2 = tl.where(valid_2d, kv_2, 0.0)
    qk += tl.dot(q_2, tl.trans(kv_2)).to(tl.float32)

    nope_fp8_3 = nope_uint8_3.to(tl.float8e4nv, bitcast=True)
    kv_3 = (nope_fp8_3.to(tl.bfloat16) * scale_bf16_3[:, None]).to(tl.bfloat16)
    kv_3 = tl.where(valid_2d, kv_3, 0.0)
    qk += tl.dot(q_3, tl.trans(kv_3)).to(tl.float32)

    nope_fp8_4 = nope_uint8_4.to(tl.float8e4nv, bitcast=True)
    kv_4 = (nope_fp8_4.to(tl.bfloat16) * scale_bf16_4[:, None]).to(tl.bfloat16)
    kv_4 = tl.where(valid_2d, kv_4, 0.0)
    qk += tl.dot(q_4, tl.trans(kv_4)).to(tl.float32)

    nope_fp8_5 = nope_uint8_5.to(tl.float8e4nv, bitcast=True)
    kv_5 = (nope_fp8_5.to(tl.bfloat16) * scale_bf16_5[:, None]).to(tl.bfloat16)
    kv_5 = tl.where(valid_2d, kv_5, 0.0)
    qk += tl.dot(q_5, tl.trans(kv_5)).to(tl.float32)

    nope_fp8_6 = nope_uint8_6.to(tl.float8e4nv, bitcast=True)
    kv_6 = (nope_fp8_6.to(tl.bfloat16) * scale_bf16_6[:, None]).to(tl.bfloat16)
    kv_6 = tl.where(valid_2d, kv_6, 0.0)
    qk += tl.dot(q_6, tl.trans(kv_6)).to(tl.float32)

    kv_7 = (rope_lo | (rope_hi << 8)).to(tl.bfloat16, bitcast=True)
    kv_7 = tl.where(valid_2d, kv_7, 0.0)
    qk += tl.dot(q_7, tl.trans(kv_7)).to(tl.float32)

    qk *= sm_scale
    qk = tl.where(valid[None, :], qk, ENG_INF)

    # update online softmax state
    m_ij = tl.max(qk, axis=1)
    m_new = tl.maximum(m_i, m_ij)
    alpha = tl.where(m_i == ENG_INF, 0.0, tl.math.exp2((m_i - m_new) * LOG2E))
    p = tl.where(qk == ENG_INF, 0.0, tl.math.exp2((qk - m_new[:, None]) * LOG2E))
    l_new = alpha * l_i + tl.sum(p, axis=1)
    p_bf16 = p.to(tl.bfloat16)

    # update accumulate
    acc_0 = acc_0 * alpha[:, None] + tl.dot(p_bf16, kv_0).to(tl.float32)
    acc_1 = acc_1 * alpha[:, None] + tl.dot(p_bf16, kv_1).to(tl.float32)
    acc_2 = acc_2 * alpha[:, None] + tl.dot(p_bf16, kv_2).to(tl.float32)
    acc_3 = acc_3 * alpha[:, None] + tl.dot(p_bf16, kv_3).to(tl.float32)
    acc_4 = acc_4 * alpha[:, None] + tl.dot(p_bf16, kv_4).to(tl.float32)
    acc_5 = acc_5 * alpha[:, None] + tl.dot(p_bf16, kv_5).to(tl.float32)
    acc_6 = acc_6 * alpha[:, None] + tl.dot(p_bf16, kv_6).to(tl.float32)
    acc_7 = acc_7 * alpha[:, None] + tl.dot(p_bf16, kv_7).to(tl.float32)

    return acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, m_new, l_new


# grid: h_q // BLOCK_H, total_token
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 32}, num_warps=4, num_stages=1),
    ],
    key=["total_tokens_bucket", "q_head_dim", "topk"],
)
@triton.jit
def sparse_fused_gather_attention_kernel(
    # Q
    q,
    q_head_dim,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    # KV cache
    kv_cache,
    kv_block_stride,
    num_of_blcoks,
    # Indices
    indices,
    indices_stride_t,
    indices_stride_topk,
    # Output
    output,
    output_stride_t,
    output_stride_h,
    output_stride_d,
    lse_ptr,
    lse_stride_t,
    lse_stride_h,
    # Topk Param
    topk_length,
    topk,
    # Other
    block_size,
    seq_len,
    sm_scale,
    total_tokens_bucket,
    HAS_TOPK_LENGTH: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # constant
    TILE_SIZE: tl.constexpr = 64
    ROPE_OFFSET: tl.constexpr = 448
    BYTES_PER_TOKEN_DATA: tl.constexpr = 576  # 448 + 64 * 2
    BYTES_PER_TOKEN_SCALE: tl.constexpr = 8
    LOG2E: tl.constexpr = 1.4426950408889634

    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_t_64 = pid_t.to(tl.int64)

    # init softmax state, accumlater
    NEG_INF = float("-inf")

    offset_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]
    q_h_masked = offset_h < q_head_dim

    m_i = tl.full([BLOCK_H], NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)

    # 512 / 8 tile = 64
    acc_0 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_1 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_2 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_3 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_4 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_5 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_6 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_7 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)

    batch_idx = pid_t // seq_len
    offset_tile = tl.arange(0, TILE_SIZE)
    # load q data
    q_stride_t_64 = q_stride_t.to(tl.int64)
    q_base = q + pid_t_64 * q_stride_t_64

    # q_n tile data shape [BLOCK_H, TILE_SIZE] = q_base + offset_h * stride_h + (tile_size * n + offset_tile) * stride_d
    q_0 = tl.load(
        q_base + offset_h[:, None] * q_stride_h + offset_tile[None, :] * q_stride_d,
        mask=q_h_masked[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_1 = tl.load(
        q_base + offset_h[:, None] * q_stride_h + (TILE_SIZE + offset_tile[None, :]) * q_stride_d,
        mask=q_h_masked[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_2 = tl.load(
        q_base
        + offset_h[:, None] * q_stride_h
        + (TILE_SIZE * 2 + offset_tile[None, :]) * q_stride_d,
        mask=q_h_masked[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_3 = tl.load(
        q_base
        + offset_h[:, None] * q_stride_h
        + (TILE_SIZE * 3 + offset_tile[None, :]) * q_stride_d,
        mask=q_h_masked[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_4 = tl.load(
        q_base
        + offset_h[:, None] * q_stride_h
        + (TILE_SIZE * 4 + offset_tile[None, :]) * q_stride_d,
        mask=q_h_masked[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_5 = tl.load(
        q_base
        + offset_h[:, None] * q_stride_h
        + (TILE_SIZE * 5 + offset_tile[None, :]) * q_stride_d,
        mask=q_h_masked[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_6 = tl.load(
        q_base
        + offset_h[:, None] * q_stride_h
        + (TILE_SIZE * 6 + offset_tile[None, :]) * q_stride_d,
        mask=q_h_masked[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_7 = tl.load(
        q_base
        + offset_h[:, None] * q_stride_h
        + (TILE_SIZE * 7 + offset_tile[None, :]) * q_stride_d,
        mask=q_h_masked[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    # gather
    if HAS_TOPK_LENGTH:
        topk_len = tl.load(topk_length + batch_idx)

    for n_start in range(0, topk, BLOCK_N):
        if not HAS_TOPK_LENGTH or n_start < topk_len:
            offset_n = n_start + tl.arange(0, BLOCK_N)  # shape: [BLOCK_N]
            mask_n = offset_n < topk  # shape [BLOCK_N]

            indices_base = indices + indices_stride_t * pid_t
            indices_data = tl.load(
                indices_base + offset_n * indices_stride_topk, mask=mask_n, other=-1
            )  # shape: [BLOCK_N]
            invalid_data = indices_data == -1
            if HAS_TOPK_LENGTH:
                invalid_data |= offset_n >= topk_len
            valid = mask_n & ~invalid_data  # shape: [BLOCK_N]
            indices_data = tl.maximum(indices_data, 0)

            block_id = indices_data // block_size  # shape: [BLOCK_N]
            slot_id = indices_data % block_size  # shape: [BLCOK_N]
            block_id_64 = block_id.to(tl.int64)
            slot_id_64 = slot_id.to(tl.int64)

            kv_block_stride_64 = tl.cast(kv_block_stride, tl.int64)
            kv_block_base = kv_cache + block_id_64 * kv_block_stride_64  # shape: [BLOCK_N]
            token_data_offset = slot_id_64 * BYTES_PER_TOKEN_DATA  # shape: [BLOCK_N]
            token_scale_offset = (
                block_size * BYTES_PER_TOKEN_DATA + slot_id_64 * BYTES_PER_TOKEN_SCALE
            )  # shape: [BLOCK_N]
            valid_2d = valid[:, None]  # shape: [BLOCK_N, 1]
            acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, m_i, l_i = process_tile(
                kv_block_base,
                token_data_offset,
                token_scale_offset,
                valid,
                valid_2d,
                q_0,
                q_1,
                q_2,
                q_3,
                q_4,
                q_5,
                q_6,
                q_7,
                acc_0,
                acc_1,
                acc_2,
                acc_3,
                acc_4,
                acc_5,
                acc_6,
                acc_7,
                m_i,
                l_i,
                offset_tile,
                sm_scale,
                TILE_SIZE,
                ROPE_OFFSET,
                LOG2E,
                BLOCK_H,
                BLOCK_N,
            )

    # lse = m_i + log l_i
    lse_value = m_i + tl.math.log2(tl.where(l_i == 0.0, 1.0, l_i)) / LOG2E
    # qkv
    # lonely q means no valid KV token
    is_lonely_q = l_i == 0  # shape [BLOCK_H]
    output_scale = tl.where(l_i == 0.0, 1.0, 1.0 / l_i)
    acc_0 = tl.where(is_lonely_q[:, None], 0.0, acc_0 * output_scale[:, None])
    acc_1 = tl.where(is_lonely_q[:, None], 0.0, acc_1 * output_scale[:, None])
    acc_2 = tl.where(is_lonely_q[:, None], 0.0, acc_2 * output_scale[:, None])
    acc_3 = tl.where(is_lonely_q[:, None], 0.0, acc_3 * output_scale[:, None])
    acc_4 = tl.where(is_lonely_q[:, None], 0.0, acc_4 * output_scale[:, None])
    acc_5 = tl.where(is_lonely_q[:, None], 0.0, acc_5 * output_scale[:, None])
    acc_6 = tl.where(is_lonely_q[:, None], 0.0, acc_6 * output_scale[:, None])
    acc_7 = tl.where(is_lonely_q[:, None], 0.0, acc_7 * output_scale[:, None])
    lse_value = tl.where(is_lonely_q, float("+inf"), lse_value)
    # store output, lse
    output_stride_t_64 = tl.cast(output_stride_t, tl.int64)
    o_base = output + output_stride_t_64 * pid_t_64

    o_0 = acc_0.to(tl.bfloat16)
    o_1 = acc_1.to(tl.bfloat16)
    o_2 = acc_2.to(tl.bfloat16)
    o_3 = acc_3.to(tl.bfloat16)
    o_4 = acc_4.to(tl.bfloat16)
    o_5 = acc_5.to(tl.bfloat16)
    o_6 = acc_6.to(tl.bfloat16)
    o_7 = acc_7.to(tl.bfloat16)

    row_ptrs = o_base + offset_h[:, None] * output_stride_h  # shape: [BLOCK_H, 1]
    tl.store(row_ptrs + offset_tile[None, :] * output_stride_d, o_0, mask=q_h_masked[:, None])
    tl.store(
        row_ptrs + (TILE_SIZE + offset_tile[None, :]) * output_stride_d,
        o_1,
        mask=q_h_masked[:, None],
    )
    tl.store(
        row_ptrs + (2 * TILE_SIZE + offset_tile[None, :]) * output_stride_d,
        o_2,
        mask=q_h_masked[:, None],
    )
    tl.store(
        row_ptrs + (3 * TILE_SIZE + offset_tile[None, :]) * output_stride_d,
        o_3,
        mask=q_h_masked[:, None],
    )
    tl.store(
        row_ptrs + (4 * TILE_SIZE + offset_tile[None, :]) * output_stride_d,
        o_4,
        mask=q_h_masked[:, None],
    )
    tl.store(
        row_ptrs + (5 * TILE_SIZE + offset_tile[None, :]) * output_stride_d,
        o_5,
        mask=q_h_masked[:, None],
    )
    tl.store(
        row_ptrs + (6 * TILE_SIZE + offset_tile[None, :]) * output_stride_d,
        o_6,
        mask=q_h_masked[:, None],
    )
    tl.store(
        row_ptrs + (7 * TILE_SIZE + offset_tile[None, :]) * output_stride_d,
        o_7,
        mask=q_h_masked[:, None],
    )

    lse_ptrs = lse_ptr + pid_t * lse_stride_t + offset_h * lse_stride_h
    tl.store(lse_ptrs, lse_value, mask=q_h_masked)
