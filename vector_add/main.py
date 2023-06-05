import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,  
    y_ptr,  
    output_ptr,  
    n_elements,  
    BLOCK_SIZE: tl.constexpr,  
):
    pid = tl.program_id(axis=0)  
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # add_kernel[grid](x, y, output)
    return output


# unit test
torch.manual_seed(0)
size = 4096
x = torch.rand(size, device="cuda")
y = torch.rand(size, device="cuda")
output_torch = x.add(y)
output_triton = add(x, y)
print(output_torch)
print(output_triton)

if torch.allclose(output_triton, output_torch, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


# Benchmark
def benchmark(size, provider):
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    y = torch.rand(size, device="cuda", dtype=torch.float32)
    quantiles = [0.2, 0.5, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x+y, quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    return ms, max_ms, min_ms


if __name__ == "__main__":
    # unit test
    torch.manual_seed(0)
    size = 4096
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    output_torch = x.add(y)
    output_triton = add(x, y)
    print(output_torch)
    print(output_triton)
    
    if torch.allclose(output_triton, output_torch, atol=1e-2, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
  
    mean_latency_torch, _, _ = benchmark(size, "torch")
    mean_latency_triton, _, _ = benchmark(size, "triton")
    print("torch:", mean_latency_torch)
    print("triton:", mean_latency_torch)



