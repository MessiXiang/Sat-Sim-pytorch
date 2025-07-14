import torch
from torch import nn


class MyModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer('a', torch.Tensor([1]))
        self.b = torch.tensor([1])

    def forward(self, x: torch.Tensor):
        self.a = x.sum()
        return self.a * 2


module = MyModule()
module.a = torch.ones(1, 5)
x = torch.rand(5, requires_grad=True)
y: torch.Tensor = module(x)
y.sum().backward()
print(x.grad)
