"""Compare the local Triton sparse decode kernel with the Torch reference."""

from __future__ import annotations

import unittest

import torch

from kernel.deepseek_v4.sparse_decode_torch import (
    torch_naive_sparse_attention_decode,
)
from kernel.deepseek_v4.sparse_decode_triton import (
    launch_sparse_fused_gather_attention_decode,
)

NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM
TILE_SIZE = 64
NUM_TILES = NOPE_DIM // TILE_SIZE
DATA_BYTES = NOPE_DIM + ROPE_DIM * 2
SCALE_BYTES = NUM_TILES + 1
BYTES_PER_TOKEN = DATA_BYTES + SCALE_BYTES


def _build_kvcache(
    num_pages: int,
    page_size: int,
    *,
    device: torch.device,
    seed: int = 0,
):
    """Build the FP8 NoPE + BF16 RoPE + UE8M0 scale page layout."""
    generator = torch.Generator(device="cpu").manual_seed(seed)

    nope_bytes = torch.randint(
        0,
        0x70,
        (num_pages, page_size, NOPE_DIM),
        generator=generator,
        dtype=torch.uint8,
    ).to(device)
    rope_bf16 = (
        torch.randn(
            (num_pages, page_size, ROPE_DIM),
            generator=generator,
            dtype=torch.float32,
        )
        .clamp(-2.0, 2.0)
        .to(device=device, dtype=torch.bfloat16)
    )
    scale_bytes = torch.randint(
        120,
        131,
        (num_pages, page_size, NUM_TILES),
        generator=generator,
        dtype=torch.uint8,
    ).to(device)

    token_data = torch.empty(
        (num_pages, page_size, DATA_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    token_data[..., :NOPE_DIM] = nope_bytes
    token_data[..., NOPE_DIM:] = rope_bf16.contiguous().view(torch.uint8)

    scale_data = torch.zeros(
        (num_pages, page_size, SCALE_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    scale_data[..., :NUM_TILES] = scale_bytes

    flat = torch.zeros(
        (num_pages, page_size * BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device=device,
    )
    flat[:, : page_size * DATA_BYTES] = token_data.reshape(num_pages, -1)
    flat[:, page_size * DATA_BYTES :] = scale_data.reshape(num_pages, -1)
    k_cache = flat.view(num_pages, page_size, 1, BYTES_PER_TOKEN).view(torch.float8_e4m3fn)

    nope_fp8 = nope_bytes.view(torch.float8_e4m3fn).float()
    scale_e8m0 = scale_bytes.view(torch.float8_e8m0fnu).float()
    nope_bf16 = (
        (
            nope_fp8.view(num_pages, page_size, NUM_TILES, TILE_SIZE)
            * scale_e8m0.view(num_pages, page_size, NUM_TILES, 1)
        )
        .view(num_pages, page_size, NOPE_DIM)
        .to(torch.bfloat16)
    )
    reference_tokens = torch.cat([nope_bf16, rope_bf16], dim=-1)
    return k_cache, reference_tokens


def _build_q_indices(
    batch_size: int,
    num_heads: int,
    topk: int,
    num_pages: int,
    page_size: int,
    *,
    device: torch.device,
    seed: int = 1,
):
    """Build BF16 queries and unique physical KV slot indices."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = (
        torch.randn(
            (batch_size, 1, num_heads, HEAD_DIM),
            generator=generator,
            dtype=torch.float32,
        )
        .clamp(-1.5, 1.5)
        .to(device=device, dtype=torch.bfloat16)
    )

    pool_size = num_pages * page_size
    indices = torch.empty(
        (batch_size, 1, topk),
        dtype=torch.int32,
        device=device,
    )
    for batch_idx in range(batch_size):
        indices[batch_idx, 0] = torch.randperm(
            pool_size,
            generator=generator,
        )[
            :topk
        ].to(device=device, dtype=torch.int32)
    return q, indices


class TestSparseDecodeTritonVsTorch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        cls.device = torch.device("cuda")

    def _run(
        self,
        batch_size: int = 1,
        num_heads: int = 16,
        topk: int = 32,
        page_size: int = 64,
        num_pages: int = 3,
        seed: int = 11,
    ):
        k_cache, _ = _build_kvcache(
            num_pages,
            page_size,
            device=self.device,
            seed=seed,
        )
        q, indices = _build_q_indices(
            batch_size,
            num_heads,
            topk,
            num_pages,
            page_size,
            device=self.device,
            seed=seed + 100,
        )
        softmax_scale = HEAD_DIM**-0.5

        ref_out, _ = torch_naive_sparse_attention_decode(
            q,
            k_cache,
            indices,
            softmax_scale,
            HEAD_DIM,
        )
        tri_out, _ = launch_sparse_fused_gather_attention_decode(
            q.reshape(batch_size, num_heads, HEAD_DIM),
            k_cache,
            indices.reshape(batch_size, topk),
            page_size,
            TopkLength=None,
            Seq_len=1,
        )
        tri_out = tri_out.reshape(batch_size, 1, num_heads, HEAD_DIM)
        torch.cuda.synchronize()

        torch.testing.assert_close(
            tri_out.float(),
            ref_out.float(),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_basic(self):
        self._run()


if __name__ == "__main__":
    unittest.main(verbosity=2)
