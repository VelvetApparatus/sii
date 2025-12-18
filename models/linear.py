import math

import torch
from torch import nn


class LinearDense(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
    ):
        super(LinearDense, self).__init__()

        # inbound
        self.in_features = in_features
        # outbound
        self.out_features = out_features

        # weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)



        self.reset_parameters()


    def reset_parameters(self):
        """
            y = W^T*x+bias
            var(y) = in_features * var(W)
            var(W) = var(y) / in_features
            var(y): 1
            var(W) = 1 / in_features

            U(-a, a) : var = a^2/3
            then:
            a ~ 1/sqrt(in_features)
        :return:
        """
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)


    def forward(self, x):
        y = x.matmul(self.weight.t())
        if self.bias is not None:
            y = y + self.bias  # broadcasting по последней оси
        return y


    def extra_repr(self) -> str:
            return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

