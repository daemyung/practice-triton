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
def add_kernel(x_ptr, y_ptr, z_ptr, size, block_size: tl.constexpr):
    offset = tl.program_id(0) * block_size

    x_block_ptr = tl.make_block_ptr(
        x_ptr, shape=(size,), strides=(1,), offsets=(offset,), block_shape=(block_size,), order=(0,)
    )
    y_block_ptr = tl.make_block_ptr(
        y_ptr, shape=(size,), strides=(1,), offsets=(offset,), block_shape=(block_size,), order=(0,)
    )

    x = tl.load(x_block_ptr, boundary_check=(0,))
    y = tl.load(y_block_ptr, boundary_check=(0,))
    z = x + y

    z_block_ptr = tl.make_block_ptr(
        z_ptr, shape=(size,), strides=(1,), offsets=(offset,), block_shape=(block_size,), order=(0,)
    )

    tl.store(z_block_ptr, z, boundary_check=(0,))


def add(x, y):
    z = torch.empty_like(x, device="cuda")
    size = z.numel()

    def grid(meta):
        return (triton.cdiv(size, meta["block_size"]),)

    add_kernel[grid](x, y, z, size, 1024)

    return z


def main():
    size = 2**16
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")

    a = add(x, y)
    b = x + y

    assert torch.allclose(a, b)


if __name__ == "__main__":
    main()
