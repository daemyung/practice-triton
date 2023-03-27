
import torch
import triton
import triton.language as tl
import tabulate

@triton.jit
def dropout_kernel(x_ptr, y_ptr, n, p, seed, BLOCK_SIZE: tl.constexpr):
    """
    드롭아웃을 수행합니다.

    :param x_ptr: 입력 텐서입니다.
    :param y_ptr: 출력 텐서입니다.
    :param n: 입력과 출력 텐서의 원소 개수입니다.
    :param p: 기준 확률입니다.
    :param seed: 랜덤 시드입니다.
    :param BLOCK_SIZE: 프로그램의 블록 크기입니다.
    """

    # 프로그램 ID를 가져옵니다.
    pid = tl.program_id(0)

    # 블록 ID를 계산합니다.
    bid = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 블록 실행 마스크를 계산합니다.
    mask = bid < n

    # 드랍아웃에 필요한 데이터를 입력 텐서로부터 읽습니다.
    x = tl.load(x_ptr + bid, mask)

    # 랜덤 확률을 계산합니다.
    probability = tl.rand(seed, bid)

    # 드롭아웃을 수행합니다.
    y = tl.where(probability > p, x / (1 - p), 0.0)

    # 계산 결과를 출력 텐서에 저장합니다.
    tl.store(y_ptr + bid, y, mask)

def dropout(x, p, seed):
    """
    드롭아웃을 수행합니다.

    :param x: 입력 텐서입니다.
    :param p: 기준 확률입니다.
    :param seed: 랜덤 시드입니다.
    :return: 드롭아웃의 결과를 반환합니다.
    """

    assert x.is_cuda
    assert x.is_contiguous()

    # 출력 텐서를 생성합니다.
    y = torch.empty_like(x)

    # 입력 텐서의 원소 개수를 가져옵니다.
    n = x.numel()

    # 동시에 실행할 프로그램의 인스턴스의 개수를 계산합니다.
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), 1, 1)

    # 커널을 실행합니다.
    dropout_kernel[grid](x, y, n, p, seed, BLOCK_SIZE=1024)

    # 계산 결과를 반환합니다.
    return y

def main():
    # 입력 텐서를 생성합니다.
    x = torch.randn(10, device='cuda')

    # 드롭아웃을 수행합니다.
    a = dropout(x, p=0.5, seed=128)
    b = dropout(x, p=0.5, seed=256)
    c = dropout(x, p=0.5, seed=512)
    d = dropout(x, p=0.5, seed=512)

    # 계산 결과를 출력합니다.
    print(tabulate.tabulate([
        ["x"] + x.tolist(),
        ["y(seed = 123)"] + a.tolist(),
        ["y(seed = 123)"] + b.tolist(),
        ["y(seed = 512)"] + c.tolist(),
        ["y(seed = 512)"] + d.tolist()
    ]))

if __name__ == '__main__':
    main()
