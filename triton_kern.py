import torch
import triton
import triton.language as tl
from torch import nn
from torch.library import triton_op


@triton.jit
def relu_kernel(x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    x = tl.load(x_ptr + idx, mask=mask)
    y = tl.where(x > 0, x, 0)
    tl.store(y_ptr + idx, y, mask=mask)


@triton_op("mylib::myrelu", mutates_args={})
def triton_relu(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    N = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    relu_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
    return y


@triton_relu.register_kernel("cpu")
def triton_relu_cpu(x):
    return torch.relu(x)


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()

    def forward(self, x):
        return triton_relu(x)


device = "cpu"
if torch.cuda.is_available():
    print("CUDA device selected")
    device = torch.cuda.current_device()

model = TestModel()
model.to(device)

ts = torch.tensor([5], device=device)

print(model.forward(ts))
