"""Compare the local Triton sparse decode kernel with the Torch reference."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import triton

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
RANDOMSEED = (11, 23, 37, 53, 71)
BENCH_RESULTS_DIR = Path(__file__).resolve().parent / "bench_results"


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


def _build_topk_length(
    batch_size: int,
    topk: int,
    *,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Build per-request valid prefix lengths for a fixed-width indices tensor."""
    if topk <= 1:
        return torch.ones(batch_size, dtype=torch.int32, device=device)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(
        low=max(1, topk // 2),
        high=topk,
        size=(batch_size,),
        generator=generator,
        dtype=torch.int32,
    ).to(device)


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
        topk_length = _build_topk_length(
            batch_size,
            topk,
            device=self.device,
            seed=seed + 200,
        )

        # The Triton path receives valid physical indices in the padded tail and
        # must ignore them using TopkLength. The Torch oracle expresses the same
        # logical input by replacing that tail with its existing -1 sentinel.
        topk_rank = torch.arange(topk, device=self.device).view(1, 1, topk)
        ref_indices = indices.masked_fill(
            topk_rank >= topk_length.view(batch_size, 1, 1),
            -1,
        )
        softmax_scale = HEAD_DIM**-0.5

        ref_out, ref_lse = torch_naive_sparse_attention_decode(
            q,
            k_cache,
            ref_indices,
            topk_length,
            softmax_scale,
            HEAD_DIM,
        )
        tri_out, tri_lse = launch_sparse_fused_gather_attention_decode(
            q.reshape(batch_size, num_heads, HEAD_DIM),
            k_cache,
            indices.reshape(batch_size, topk),
            page_size,
            TopkLength=topk_length,
            Seq_len=1,
        )
        tri_out = tri_out.reshape(batch_size, 1, num_heads, HEAD_DIM)
        tri_lse = tri_lse.reshape(batch_size, 1, num_heads).permute(0, 2, 1)
        torch.cuda.synchronize()

        torch.testing.assert_close(
            tri_out.float(),
            ref_out.float(),
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            tri_lse.float(),
            ref_lse.float(),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_basic(self):
        for seed in RANDOMSEED:
            with self.subTest(seed=seed):
                self._run(seed=seed)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["context_len"],
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "-")],
        xlabel="Context length",
        ylabel="Latency (us)",
        plot_name="sparse-decode-context-length-topk128-latency",
        args={
            "batch_size": 1,
            "num_heads": 16,
            "page_size": 64,
            "topk": 128,
            "seed": RANDOMSEED[0],
        },
    )
)
@torch.inference_mode()
def benchmark_latency(
    context_len: int,
    provider: str,
    batch_size: int,
    num_heads: int,
    page_size: int,
    topk: int,
    seed: int,
):
    """Benchmark fixed-topk sparse decode while growing the KV context."""
    device = torch.device("cuda")
    if context_len < topk:
        raise ValueError(f"context_len={context_len} must be >= topk={topk}")
    num_pages = (context_len + page_size - 1) // page_size

    k_cache, _ = _build_kvcache(
        num_pages,
        page_size,
        device=device,
        seed=seed,
    )
    q, indices = _build_q_indices(
        batch_size,
        num_heads,
        topk,
        num_pages,
        page_size,
        device=device,
        seed=seed + 100,
    )
    topk_length = torch.full(
        (batch_size,),
        topk,
        dtype=torch.int32,
        device=device,
    )

    softmax_scale = HEAD_DIM**-0.5

    q_flat = q.reshape(batch_size, num_heads, HEAD_DIM)
    indices_flat = indices.reshape(batch_size, topk)

    if provider == "torch":

        def torch_fn():
            return torch_naive_sparse_attention_decode(
                q,
                k_cache,
                indices,
                topk_length,
                softmax_scale,
                HEAD_DIM,
            )

        benchmark_fn = torch_fn

    elif provider == "triton":

        def triton_fn():
            return launch_sparse_fused_gather_attention_decode(
                q_flat,
                k_cache,
                indices_flat,
                page_size,
                TopkLength=topk_length,
                Seq_len=1,
            )

        benchmark_fn = triton_fn

    else:
        raise ValueError(f"unknown provider: {provider}")

    timings = triton.testing.do_bench(
        benchmark_fn,
        warmup=100,
        rep=500,
        quantiles=[0.5, 0.2, 0.8],
    )
    if not isinstance(timings, (list, tuple)) or len(timings) != 3:
        raise TypeError(f"unexpected do_bench result: {timings!r}")

    median_ms, p20_ms, p80_ms = (float(value) for value in timings)
    return median_ms * 1000, p20_ms * 1000, p80_ms * 1000


if __name__ == "__main__":
    if "--bench" in sys.argv:
        BENCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        benchmark_latency.run(
            show_plots=False,
            print_data=False,
            save_path=str(BENCH_RESULTS_DIR),
        )
    else:
        unittest.main(verbosity=2)
