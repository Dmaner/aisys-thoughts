import os
from pathlib import Path

os.environ["CUDA_HOME"] = "/usr/local/cuda-13.0"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load


# 1. 编译并加载 CUDA 自定义算子；TORCH_LIBRARY 会把它注册到 torch.ops。
load(
    name="online_softmax_ops_ext",
    sources=[str(Path(__file__).parent / "online_softmax.cu")],
    extra_cuda_cflags=["-O3", "-lineinfo", "-arch=sm_89"],
    with_cuda=True,
    is_python_module=False,
)

# 2. 直接传入 PyTorch CUDA Tensor。
torch.manual_seed(42)
x = torch.randn(8, 4096, device="cuda", dtype=torch.float32)
actual = torch.ops.online_softmax_ops.forward(x)

# 3. 用 PyTorch FP64 Softmax 生成 golden。
golden = F.softmax(x.double(), dim=-1)
error = (actual.double() - golden).abs()
max_abs_error = error.max().item()
max_relative_error = (error / golden.abs()).max().item()

print(f"max absolute error: {max_abs_error:.3e}")
print(f"max relative error: {max_relative_error:.3e}")

assert max_relative_error < 1.0e-4
print("PASS")
