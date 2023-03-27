import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32 * 1}, num_warps=32, num_stages=4),
        triton.Config({'BLOCK_SIZE': 32 * 2}, num_warps=32, num_stages=4),
        triton.Config({'BLOCK_SIZE': 32 * 4}, num_warps=32, num_stages=4),
        triton.Config({'BLOCK_SIZE': 32 * 8}, num_warps=32, num_stages=4),
        triton.Config({'BLOCK_SIZE': 32 * 16}, num_warps=32, num_stages=4),
        triton.Config({'BLOCK_SIZE': 32 * 32}, num_warps=32, num_stages=4),
        triton.Config({'BLOCK_SIZE': 32 * 64}, num_warps=32, num_stages=4)
    ],
    key=['x_ptr']
)
@triton.jit
def layer_norm_kernel(x_ptr, y_ptr, stride, n, eps,
                      BLOCK_SIZE: tl.constexpr):
    """
    레이어 정규화를 계산합니다.

    :param x_ptr: 입력 텐서입니다.
    :param y_ptr: 출력 텐서입니다.
    :param stride: 입력 텐서의 스트라이드입니다.
    :param n: 원소의 개수입니다.
    :param eps: 입실론입니다.
    :param BLOCK_SIZE: 프로그램의 블록 크기입니다.
    """

    # 프로그램 ID를 가져옵니다.
    pid = tl.program_id(0)

    # 프로그램 ID에 해당하는 포인터로 이동합니다.
    x_ptr += pid * stride
    y_ptr += pid * stride

    # 임시 변수를 생성합니다.
    tmp = tl.zeros([BLOCK_SIZE], dtype=tl.float16)

    # 평균을 계산하기 위해 블록 단위로 계산을 합니다.
    for offset in range(0, n, BLOCK_SIZE):
        bid = offset + tl.arange(0, BLOCK_SIZE)
        mask = bid < n
        x = tl.load(x_ptr + bid, mask, other=0.0, eviction_policy="evict_last")
        tmp += x

    # 평균을 계산합니다.
    mean = tl.sum(tmp, 0) / n

    # 임시 변수를 생성합니다.
    tmp = tl.zeros([BLOCK_SIZE], dtype=tl.float16)

    # 분산을 계산하기 위해 블록 단위로 계산을 합니다.
    for offset in range(0, n, BLOCK_SIZE):
        bid = offset + tl.arange(0, BLOCK_SIZE)
        mask = bid < n
        x = tl.load(x_ptr + bid, mask, other=0.0, eviction_policy="evict_last")
        x = tl.where(bid < n, x - mean, 0.).to(tl.float16)
        tmp += x * x

    # 분산을 계산합니다.
    var = tl.sum(tmp, 0) / n

    # 표준 편차를 계산합니다.
    std = tl.sqrt(var + eps)

    # 레이어 정규화를 블록 단위로 계산합니다.
    for offset in range(0, n, BLOCK_SIZE):
        bid = offset + tl.arange(0, BLOCK_SIZE)
        mask = bid < n
        x = tl.load(x_ptr + bid, mask, other=0.0, eviction_policy="evict_last")
        tl.store(y_ptr + bid, (x - mean) / std, mask)


def layer_normal(x, eps: float = 1e-05):
    """
    레이어 정규화를 계산합니다.

    :param x: 입력 텐서입니다.
    :param eps: 입실론입니다.
    :return: 레이어 정규화 결과를 반환합니다.
    """
    # 출력 텐서를 생성합니다.
    y = torch.zeros_like(x)

    # 커널을 실행합니다.
    layer_norm_kernel[(x.shape[0], )](x, y, x.stride(0), x.shape[1], eps)

    # 결과를 반환합니다.
    return y

def main():
    # 입력 텐서를 생성합니다.
    x = torch.randn((1024, 2096), device='cuda', dtype=torch.float16)

    # PyTorch로 계산합니다.
    a = torch.layer_norm(x, x.size()[1:])

    # Triton으로 계산합니다.
    b = layer_normal(x)

    # 계산 결과가 같은지 출력합니다.
    print(f'Do Triton and Pytorch produce same results? {triton.testing.allclose(a, b)}.')

if __name__ == '__main__':
    main()










