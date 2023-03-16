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

@torch.jit.script
def softmax(x):
    """
    각 행의 소프트맥스를 계산합니다.

    :param x: 입력 텐서입니다.
    :return: 소프트맥스의 결과를 반환합니다.
    """

    assert x.dim() == 2

    # 각 열의 최댓값을 계산합니다.
    y = torch.max(x, dim=1)[0]

    # 1차원 텐서를 2차원 텐서로 변경합니다.
    y = y[:, None]

    # 소프트맥스에 필요한 값들을 계산합니다.
    a = x - y
    b = torch.exp(a)
    c = torch.sum(b, dim=1)

    # 1차원 텐서를 2차원 텐서로 변경합니다.
    c = c[:, None]

    # 소프트맥스를 계산합니다.
    d = b / c

    # 계산 결과를 반환합니다.
    return d


@triton.jit
def softmax_kernel(x_ptr, y_ptr, stride, col, BLOCK_SIZE: tl.constexpr):
    """
    각 행의 소프트맥스를 계산합니다.

    :param x_ptr: 입력 텐서입니다.
    :param y_ptr: 출력 텐서입니다.
    :param stride: 입력 텐서의 스트라이드입니다.
    :param col: 행렬의 열 개수입니다.
    :param BLOCK_SIZE: 프로그램의 블록 크기입니다.
    """

    # 행렬의 행을 가져옵니다. 커널 실행시 행의 개수만큼 블럭을 생성하기 때문에 프로그램 ID가 행렬의 행입니다.
    row = tl.program_id(0)

    # 메모리 오프셋을 계산합니다.
    offset = row * stride

    # 쓰레드 ID와 실행 마스크를 계산합니다.
    tid = tl.arange(0, BLOCK_SIZE)
    exe = tid < col
    tid = offset + tid

    # 계산에 필요한 데이터를 입력 텐서로부터 읽습니다.
    x = tl.load(x_ptr + tid, mask=exe, other=-float('inf'))

    # 최대값을 계산합니다.
    y = tl.max(x, axis=0)

    #소프트맥스를 계산합니다.
    a = x - y
    b = tl.exp(a)
    c = tl.sum(b, axis=0)
    d = b / c

    # 계산 결과를 출력 텐서에 저장합니다.
    tl.store(y_ptr + tid, d, mask=exe)

def fused_softmax(x):
    """
    각 행의 소프트맥스를 계산합니다.

    :param x: 입력 텐서입니다.
    :return: 소프트맥스의 결과를 반환합니다.
    """

    assert x.dim() == 2 and x.is_cuda

    # 텐서의 행과 열의 개수를 가져옵니다.
    row, col = x.shape

    # 출력 텐서를 생성합니다.
    y = torch.empty_like(x)

    # 커널 메터 정보를 정의합니다.
    meta = {
        'BLOCK_SIZE': triton.next_power_of_2(col),
        'num_warps': 32
    }

    # 소프트맥스를 계산하기 위해 커널을 실행합니다.
    softmax_kernel[(row, )](x, y, x.stride(0), col, **meta)

    # 계산 결과를 반환합니다.
    return y

def main():
    # 입력 텐서를 생성합니다.
    x = torch.randn(256, 4096, device='cuda')

    # PyTorch로 소프트맥스를 계산합니다.
    a = torch.softmax(x, axis=1)

    # PyTorch Jit으로 소프트맥스를 계산합니다.
    b = softmax(x)

    # Triton으로 소프트맥스를 계산합니다.
    c = fused_softmax(x)

    # 입실론 값을 정의합니다.
    eps = 1e-8

    # 세 계산 결과의 차이를 출력합니다.
    print(f'Error between PyTorch     and PyTorch Jit is'
          f' {torch.sum(torch.where(torch.abs(a - b) <= eps, 0, torch.abs(a - b)))}.')
    print(f'Error between PyTorch Jit and Triton      is'
          f' {torch.sum(torch.where(torch.abs(b - c) <= eps, 0, torch.abs(b - c)))}.')
    print(f'Error between Triton      and PyTorch     is'
          f' {torch.sum(torch.where(torch.abs(c - a) <= eps, 0, torch.abs(c - a)))}.')

if __name__ == '__main__':
    main()
