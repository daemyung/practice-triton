# MIT License
#
# Copyright (c) 2023 Daemyung Jang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    m_size,
    k_size,
    n_size,
    m_block_size: tl.constexpr,
    k_block_size: tl.constexpr,
    n_block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(n_size, n_block_size)
    m_block = pid // num_n_blocks
    n_block = pid % num_n_blocks

    m_offsets = tl.arange(0, m_block_size) + m_block * m_block_size
    n_offsets = tl.arange(0, n_block_size) + n_block * n_block_size
    k_offsets = tl.arange(0, k_block_size)

    x_ptrs = x_ptr + m_offsets[:, None] * k_size + k_offsets[None, :]
    y_ptrs = y_ptr + k_offsets[:, None] * n_size + n_offsets[None, :]
    z_ptrs = z_ptr + m_offsets[:, None] * n_size + n_offsets[None, :]

    z = tl.zeros((m_block_size, n_block_size), tl.float32)

    for _ in range(0, k_size, k_block_size):
        x = tl.load(x_ptrs)
        y = tl.load(y_ptrs)
        z += tl.dot(x, y, allow_tf32=False)

        x_ptrs += k_block_size
        y_ptrs += k_block_size * n_size

    tl.store(z_ptrs, z)


def matmul(x, y):
    m_size, k_size = x.shape
    _, n_size = y.shape
    z = torch.empty(m_size, n_size, device="cuda")

    def grid(meta):
        return (triton.cdiv(m_size, meta["m_block_size"]) * triton.cdiv(n_size, meta["n_block_size"]),)

    matmul_kernel[grid](x, y, z, m_size, k_size, n_size, m_size, k_size, n_size)

    return z


def main():
    x = torch.randn(16, 16, device="cuda")
    y = torch.randn(16, 16, device="cuda")

    a = matmul(x, y)
    b = torch.matmul(x, y)

    assert torch.allclose(a, b)


if __name__ == "__main__":
    main()
