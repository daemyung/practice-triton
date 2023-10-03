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
def combine_add(a, b):
    return a + b


@triton.jit
def sum_kernel(y_ptr, x_ptr, size, block_size: tl.constexpr):
    offsets = tl.arange(0, block_size)
    mask = offsets < size

    x = tl.load(x_ptr + offsets, mask)
    y = tl.reduce(x, 0, combine_add)
    tl.store(y_ptr, y)


def sum(x):
    size = x.numel()
    y = torch.empty(1, device="cuda")

    def grid(meta):
        return (1,)

    sum_kernel[grid](y, x, size, triton.next_power_of_2(size))

    return y


def main():
    x = torch.randn(1024, device="cuda")

    a = sum(x)
    b = torch.sum(x)

    assert torch.allclose(a, b)


if __name__ == "__main__":
    main()
