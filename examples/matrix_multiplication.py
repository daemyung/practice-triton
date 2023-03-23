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

BLOCK_K_MULTIPLICAND = 32
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': BLOCK_K_MULTIPLICAND, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=32),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': BLOCK_K_MULTIPLICAND, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=16),
    ],
    key=['m', 'n', 'k']
)
@triton.jit
def matmul_kernel(x_ptr, stride_xm, stride_xk,
                  y_ptr, stride_yk, stride_yn,
                  z_ptr, stride_zm, stride_zn,
                  m, n, k,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                  GROUP_SIZE_M: tl.constexpr,):
    """
    두 개의 입력을 받아 곱한 후 출력에 저장합니다.

    :param x_ptr: 첫 번째 입력입니다.
    :param stride_xm: 첫 번째 입력의 열 스트라이드입니다.
    :param stride_xk: 첫 번째 입력의 행 스트라이드입니다.
    :param y_ptr: 두 번째 입력입니다.
    :param stride_yk: 두 번째 입력의 열 스트라이드입니다.
    :param stride_yn: 두 번째 입력의 행 스트라이드입니다.
    :param z_ptr: 곱셈의 결과가 저장될 출력입니다.
    :param stride_zm: 출력의 행 스트라이드입니다.
    :param stride_zn: 출력의 열 스트라이드입니다.
    :param m: 첫 번째 입력의 행의 크기입니다.
    :param n: 두 번쨰 입력의 열의 크기입니다.
    :param k: 첫 번째 입력의 열 그리고 두 번째 입력의 행의 크기입니다.
    :param BLOCK_SIZE_M: 프로그램의 블록 크기입니다.
    :param BLOCK_SIZE_N: 프로그램의 블록 크기입니다.
    :param BLOCK_SIZE_K: 프로그램의 블록 크기입니다.
    :param GROUP_SIZE_M: 그룹의 크기입니다.
    """
    # 프로그램 ID를 가져옵니다.
    pid = tl.program_id(0)

    # 행렬 x의 m행의 PID 개수를 계산합니다.
    num_pid_m = tl.cdiv(m, BLOCK_SIZE_M)

    # 행렬 y의 n행의 PID 개수를 계산합니다.
    num_pid_n = tl.cdiv(n, BLOCK_SIZE_N)

    # 그룹 1개의 PID 개수를 계산합니다.
    num_pid_g = GROUP_SIZE_M * num_pid_n

    # 그룹의 ID를 계산합니다.
    gid = pid // num_pid_g

    # 그룹의 m행의 첫번째 PID를 계산합니다.
    first_pid_m = gid * GROUP_SIZE_M

    # num_pid_m가 GROUP_SIZE_M으로 나뉘지 않는다면 마지막 그룹의 크기는 GROUP_SIZE_M보다 작습니다.
    # 그러므로 올바른 그룹 크기를 계산합니다.
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    # m행의 PID를 계산합니다.
    pid_m = first_pid_m + (pid % group_size_m)

    # n열의 PID를 계산합니다.
    pid_n = pid % num_pid_g

    # m, n, k의 오프셋을 계산합니다.
    offset_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_k = tl.arange(0, BLOCK_SIZE_K)

    # x에서 데이터를 읽을 공간([BLOCK_SIZE_M, BLOCK_SIZE_K])의 포인터를 계산합니다.
    a_ptr = x_ptr + (offset_xm[:, None] * stride_xm + offset_k[None, :] * stride_xk)

    # y에서 데이터를 읽을 공간([BLOCK_SIZE_K, BLOCK_SIZE_N])의 포인터를 계산합니다.
    b_ptr = y_ptr + (offset_k[:, None] * stride_yk + offset_yn[None, :] * stride_yn)

    # 계산 결과를 저장할 임시 텐서를 생성합니다.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 행렬곱 계산을 위한 반복문입니다.
    for i in range(0, k, BLOCK_SIZE_K):
        # 행렬곱에 필요한 데이터를 입력 텐서로부터 읽습니다.
        a = tl.load(a_ptr)
        b = tl.load(b_ptr)

        # 행렬곱의 일부를 계산합니다.
        accumulator += tl.dot(a, b)

        # 행의 방향으로 BLOCK_SIZE_K만큼 이동합니다.
        a_ptr += BLOCK_SIZE_K * stride_xk

        # 열의 방향으로 BLOCK_SIZE_K만큼 이동합니다.
        b_ptr += BLOCK_SIZE_K * stride_yk

    # float32에서 float16으로 변환합니다.
    c = accumulator.to(tl.float16)

    # m, n의 오프셋을 계산합니다.
    offset_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # 데이터를 쓸 공간([BLOCK_SIZE_M, BLOCK_SIZE_N])의 포인터를 계산합니다.
    c_ptr = z_ptr + offset_cm[:, None] * stride_zm + offset_cn[None, :] * stride_zn

    # 실행이 필요한 블록 마스크를 계산합니다.
    mask = (offset_cm[:, None] < m) & (offset_cn[None, :] < n)

    # 행렬곱의 결과를 출력 텐서에 저장합니다.
    tl.store(c_ptr, c, mask)

def matmul(x, y):
    """
    두 개의 입력 텐서를 받아 곱한 결과를 반환합니다.

    :param x: 첫 번째 입력 텐서입니다.
    :param y: 두 번째 입력 텐서입니다.
    :return: 행렬곱의 결과를 반환합니다.
    """
    assert x.is_cuda and y.is_cuda
    assert x.shape[1] == y.shape[0]
    assert x.is_contiguous()
    assert y.is_contiguous()

    # 행렬곱에 필요한 정보를 두 행렬에서 가져옵니다.
    m, k = x.shape
    _, n = y.shape

    # 메모리 경계 검사를 하지 않기 때문에 BLOCK_K_MULTIPLICAND로 반드시 나눠져야 합니다.
    assert k % BLOCK_K_MULTIPLICAND == 0

    # 출력 텐서를 생성합니다.
    z = torch.empty((m, n), device='cuda', dtype=torch.float16)

    # 동시에 실행해야하는 인스턴스의 개수를 계산합니다.
    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_SIZE_M']) * triton.cdiv(n, meta['BLOCK_SIZE_N']), )

    # 커널을 실행하여 행렬곱을 계산합니다.
    matmul_kernel[grid](x, x.stride(0), x.stride(1),
                        y, y.stride(0), y.stride(1),
                        z, z.stride(0), z.stride(1),
                        m, n, k)

    # 계산 결과를 반환합니다.
    return z

def main():
    # 입력 텐서를 생성합니다.
    x = torch.randn(512, 512, device='cuda', dtype=torch.float16)
    y = torch.randn(512, 512, device='cuda', dtype=torch.float16)

    # Triton으로 행렬곱을 계산합니다.
    c = matmul(x, y)

    # PyTorch로 행렬곱을 계산합니다.
    d = torch.matmul(x, y)

    # 계산 결과가 같은지 출력합니다.
    print(f'Do Triton and Pytorch produce same results? {triton.testing.allclose(c, d)}.')

if __name__ == '__main__':
    main()

