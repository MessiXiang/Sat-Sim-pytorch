import torch


class A(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.module = torch.nn.Linear(5, 10)
