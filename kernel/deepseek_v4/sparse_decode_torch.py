# Query: [B, S, H, D] = [B, S, H, 512]
# KV_Cache: [num_block, block_size, H, D] = [num_pages, block_size, 1, 512]
# Indices: [B, S, K] = [B, S, topk]

import torch

DEVICE = torch.device("cuda")
ENG_INF = float("-inf")
NOPE_DIM = 448
NOPE_BYTES_OFFSET = 448
ROPE_DIM = 64
ROPE_BYTES_OFFSET = 128
SCLAE_DIM = 7
SCALE_BYTES_OFFSET = 8  # 7 scale + 1 padding
NOPE_ROPE_DIM = 512
NOPE_ROPE_BYTES_OFFSET = NOPE_BYTES_OFFSET + ROPE_BYTES_OFFSET
TILE_SIZE = 64
NUM_TILES = NOPE_DIM // TILE_SIZE  # 7


def _gather_and_dequant(kv_cache: torch.Tensor, indices: torch.Tensor, block_size):
    """
    kv cache: [num_blocks, block_size, 1, bytes_per_token]
    indices: [B, S, top_k]
    """
    flat_indices = indices.reshape(-1)  # (N,)
    total_token = flat_indices.shape[0]

    selected_page = flat_indices // block_size  # (N,)
    selected_slot = flat_indices % block_size  # (N,)

    num_blocks = kv_cache.shape[0]
    kv_uint8 = kv_cache.view(torch.uint8)
    page_bytes = kv_uint8.stride(0)

    # Expose every physical page as one byte row, including any page padding.
    blocks = kv_uint8.as_strided(
        (num_blocks, page_bytes),
        (page_bytes, 1),
    )

    # nope
    slot_nope_base = selected_slot * NOPE_ROPE_BYTES_OFFSET
    selected_nope_slot = slot_nope_base[:, None] + torch.arange(
        NOPE_DIM, device=kv_cache.device, dtype=torch.long
    )
    selected_nope = blocks[selected_page[:, None], selected_nope_slot]
    # rope
    slot_rope_base = slot_nope_base + NOPE_BYTES_OFFSET
    selected_rope_slot = slot_rope_base[:, None] + torch.arange(
        ROPE_DIM * 2, device=kv_cache.device, dtype=torch.long
    )
    selected_rope = blocks[selected_page[:, None], selected_rope_slot]
    # quantize scale
    slot_scale_base = selected_slot * SCALE_BYTES_OFFSET + block_size * NOPE_ROPE_BYTES_OFFSET
    selected_scale_slot = slot_scale_base[:, None] + torch.arange(
        SCLAE_DIM, device=kv_cache.device, dtype=torch.long
    )
    selected_scale = blocks[selected_page[:, None], selected_scale_slot]

    nope_fp8 = selected_nope.view(torch.float8_e4m3fn)
    rope_bf16 = selected_rope.contiguous().view(torch.bfloat16)
    scale_e8m0 = selected_scale.view(torch.float8_e8m0fnu)

    result = torch.empty(
        total_token,
        NOPE_ROPE_DIM,
        device=kv_cache.device,
        dtype=torch.bfloat16,
    )
    result[:, :NOPE_DIM] = (
        (
            nope_fp8.view(total_token, NUM_TILES, TILE_SIZE).float()  # (N, 7, 64)
            * scale_e8m0.view(total_token, NUM_TILES, 1).float()
        )
        .view(total_token, NOPE_DIM)
        .to(torch.bfloat16)
    )
    result[:, NOPE_DIM:] = rope_bf16
    return result.reshape(*indices.shape, NOPE_ROPE_DIM)  # (N, topk, 512)


def torch_naive_sparse_attention_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale,
    hidden_dim,
):
    D_q = q.shape[-1]
    block_size = kv_cache.shape[1]

    invalid_mask = indices < 0
    safe_indices = indices.clamp(min=0)

    # gather kv [B, S, H, D]
    gathered_kv = _gather_and_dequant(kv_cache, safe_indices, block_size)

    # dequantize
    q_f = q.float()
    kv_f = gathered_kv.float()
    kv_d = kv_f.shape[-1]
    if D_q != kv_d:
        q_f = q_f[..., :kv_d]

    scores = torch.einsum("bshd,bstd->bsht", q_f, kv_f) * softmax_scale
    scores.masked_fill_(invalid_mask.unsqueeze(2).expand_as(scores), ENG_INF)

    lse = torch.logsumexp(scores, dim=-1)

    lonely = lse == ENG_INF
    lse[lonely] = ENG_INF
    weights = torch.exp(scores - lse.unsqueeze(-1))
    out = torch.einsum("bsht,bstv->bshv", weights, kv_f[..., :hidden_dim])
    out[lonely.unsqueeze(-1).expand_as(out)] = 0.0

    return out.to(torch.bfloat16), lse.permute(0, 2, 1)
