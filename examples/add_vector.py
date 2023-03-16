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
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    두 개의 입력을 받아 더한 후 출력에 저장합니다.

    :param x_ptr: 첫 번째 입력입니다.
    :param y_ptr: 두 번째 입력입니다.
    :param z_ptr: 덧셈의 결과가 저장될 출력입니다.
    :param size: 입력과 출력의 크기입니다.
    :param BLOCK_SIZE: 프로그램의 블록 크기입니다.
    """

    # 프로그램 ID를 가져옵니다.
    pid = tl.program_id(axis=0)

    # 프로그램 ID와 BLOCK_SIZE를 이용하여 프로그램의 첫 번째 블록 인덱스를 계산합니다.
    offset = pid * BLOCK_SIZE

    # 프로그램에 해당하는 블록 ID를 생성합니다.
    bid = offset + tl.arange(0, BLOCK_SIZE)

    # 실행이 필요한 블록 마스크를 계산합니다.
    exe = bid < size

    # 프로그램에서 계산에 필요한 데이터를 입력 텐서로부터 읽습니다.
    x = tl.load(x_ptr + bid, mask=exe)
    y = tl.load(y_ptr + bid, mask=exe)

    # 덧셈을 합니다.
    z = x + y

    # 덧셈 결과를 출력 텐서에 저장합니다.
    tl.store(z_ptr + bid, z, mask=exe)

def add(x: torch.Tensor, y: torch.Tensor):
    """
    두 개의 입력 텐서를 받아 더한 결과를 반환합니다.

    :param x: 첫 번째 입력 텐서입니다.
    :param y: 두 번째 입력 텐서입니다.
    :return: 덧셈의 결과를 반환합니다.
    """
    assert x.is_cuda and y.is_cuda

    # 출력 텐서를 생성합니다.
    z = torch.empty_like(x, device='cuda')
    n = z.numel()

    # SPMD(Single Program Multiple Data)를 따르기 위해 동시에 실행해야하는 인스턴스의 개수를 계산합니다.
    grid = lambda meta: (max(1, triton.cdiv(n, meta['BLOCK_SIZE'])), )

    # 커널을 실행하여 덧셈을 계산합니다.
    add_kernel[grid](x, y, z, n, BLOCK_SIZE=512)

    # 계산 결과를 반환합니다.
    return z

def main():
    # 텐서의 크기를 계산합니다.
    size = 2 ** 16

    # 입력 텐서를 생성합니다.
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')

    # Triton으로 덧셈을 합니다.
    a = add(x, y)

    # PyTorch로 덧셈을 합니다.
    b = x + y

    # 두 계산 결과의 차이를 출력합니다.
    print(f'Error between Triton and Pytorch is {torch.sum(torch.abs(a - b))}.')

if __name__ == '__main__':
    main()
