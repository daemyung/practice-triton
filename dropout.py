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
import torch.autograd as autograd
import torch.random as random
import torch.nn as nn
import triton
import triton.language as tl


class DropoutKernel:
    @staticmethod
    @triton.jit
    def forward(output_ptr, input_ptr, size, p, seed, block_size: tl.constexpr):
        pid = tl.program_id(0)
        offset = pid * block_size

        input_block_ptr = tl.make_block_ptr(
            input_ptr, shape=(size,), strides=(1,), offsets=(offset,), block_shape=(block_size,), order=(0,)
        )
        output_block_ptr = tl.make_block_ptr(
            output_ptr, shape=(size,), strides=(1,), offsets=(offset,), block_shape=(block_size,), order=(0,)
        )

        offsets = tl.arange(0, block_size) + offset
        random_values = tl.rand(seed, offsets)
        condition = random_values > p
        input = tl.load(input_block_ptr, boundary_check=(0,))
        output = tl.where(condition, input * (1 / (1 - p)), 0.0)
        tl.store(output_block_ptr, output, boundary_check=(0,))

    @staticmethod
    @triton.jit
    def backward(grad_input_ptr, grad_output_ptr, output_ptr, size, p, block_size: tl.constexpr):
        pid = tl.program_id(0)
        offset = pid * block_size

        grad_input_block_ptr = tl.make_block_ptr(
            grad_input_ptr, shape=(size,), strides=(1,), offsets=(offset,), block_shape=(block_size,), order=(0,)
        )
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr, shape=(size,), strides=(1,), offsets=(offset,), block_shape=(block_size,), order=(0,)
        )
        output_block_ptr = tl.make_block_ptr(
            output_ptr, shape=(size,), strides=(1,), offsets=(offset,), block_shape=(block_size,), order=(0,)
        )

        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,))
        output = tl.load(output_block_ptr, boundary_check=(0,))
        condition = output > 0.0
        grad_input = tl.where(condition, grad_output * (1 / (1 - p)), 0.0)
        tl.store(grad_input_block_ptr, grad_input, boundary_check=(0,))


class DropoutFunction(autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        input, p, training = args
        output = torch.empty_like(input)
        size = input.numel()
        block_size = triton.next_power_of_2(input.shape[-1])

        def grid(meta):
            return (triton.cdiv(size, meta["block_size"]),)

        DropoutKernel.forward[grid](output, input, size, p, random.seed(), block_size)

        ctx.save_for_backward(output)
        ctx.p = p

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        (grad_output,) = grad_outputs
        (output,) = ctx.saved_tensors
        grad_input = torch.empty_like(grad_output)
        size = grad_input.numel()
        block_size = triton.next_power_of_2(grad_input.shape[-1])

        def grid(meta):
            return (triton.cdiv(size, meta["block_size"]),)

        DropoutKernel.backward[grid](grad_input, grad_output, output, size, ctx.p, block_size)

        return grad_input, None, None


def dropout(input, p=0.5, training=True):
    if training:
        return DropoutFunction.apply(input, p, training)
    else:
        return input


class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, input):
        return dropout(input, self.p, self.training)


def main():
    input = torch.rand(6, device="cuda", requires_grad=True)
    p = 0.4
    output = dropout(input, p)
    grad_output = torch.ones_like(output)
    output.backward(grad_output)

    print(f"input     : {input.data}")
    print(f"output    : {output.data}")
    print(f"grad_input: {input.grad.data}")


if __name__ == "__main__":
    main()
