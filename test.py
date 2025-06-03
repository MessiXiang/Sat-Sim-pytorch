from torch import nn


class A(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 10)

    def forward(self, x):
        return self.fc(x)
