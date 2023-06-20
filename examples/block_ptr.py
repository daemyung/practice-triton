# MIT License
#
# Copyright (c) 2023 daemyung jang
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


@triton.jit
def write_block_ptr(out_ptr, block_size: triton.language.constexpr):
    ver = triton.language.arange(0, block_size)
    hor = triton.language.arange(0, block_size) * block_size
    blk = hor[None, :] + ver[:, None]
    blk = triton.language.view(blk, [block_size * block_size])

    wrt = triton.language.arange(0, block_size * block_size)
    triton.language.store(out_ptr + wrt, blk)


def main():
    out_size = 4
    out = torch.zeros(out_size * out_size, device='cuda')

    write_block_ptr[(1,)](out, out_size)
    print(f'\nblock pointers:\n{out}')


if __name__ == '__main__':
    main()
