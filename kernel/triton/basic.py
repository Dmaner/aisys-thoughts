import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # blockIdx.x
    pid = tl.program_id(axis=0)
    # blockIdx.x * BLOCK_SIZE
    block_start = pid * BLOCK_SIZE

    # for(int i = threadIdx.x; i < n_elements; i += blockDim.x)
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


# vector add
def add(
    x: torch.Tensor,
    y: torch.Tensor,
):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](
        x, y, output, n_elements, BLOCK_SIZE=1024
    )  # pyright: ignore[reportArgumentType]
    return output


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["triton", "torch"],  # Possible values for `line_arg`.
        line_names=["Triton", "Torch"],  # Label name for the lines.
        styles=[("blue", "-"), ("green", "-")],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="vector-add-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def vector_add_benchmark(size, provider):

    # vector add
    vector_add_x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    vector_add_y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    vector_add_quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        vector_add_timings = triton.testing.do_bench(
            lambda: vector_add_x + vector_add_y, quantiles=vector_add_quantiles
        )
    if provider == "triton":
        vector_add_timings = triton.testing.do_bench(
            lambda: add(vector_add_x, vector_add_y), quantiles=vector_add_quantiles
        )

    assert isinstance(vector_add_timings, list)
    assert len(vector_add_timings) == 3
    median_ms, p20_ms, p80_ms = vector_add_timings

    # three data transfer: load x, load y, store z = x + y
    bytes_moved = 3 * vector_add_x.numel() * vector_add_x.element_size()
    gbps = lambda ms: bytes_moved / (ms * 1e-3) / 1e9
    return gbps(median_ms), gbps(p20_ms), gbps(p80_ms)


if __name__ == "__main__":
    vector_add_benchmark.run(  # pyright: ignore[reportUndefinedVariable]
        print_data=True,
        show_plots=True,
        save_path="./bench_results",
    )
