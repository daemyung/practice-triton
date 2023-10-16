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

from typing import Any

import torch


class Mul(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        a, b = args
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        a, b = ctx.saved_tensors
        return b, a


mul = Mul.apply


def main():
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0)
    c = mul(a, b)
    print(f"{a} * {b} = {c}")
    c.backward()
    print(f"a's gradient is {a.grad}")


if __name__ == "__main__":
    main()
