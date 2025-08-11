import torch


class A(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_module = torch.nn.Linear(10, 2)
        self.direct_module = torch.nn.Linear(10, 2)

    @property
    def custom_module(self) -> torch.nn.Module:
        return self._custom_module


a = A()
for name, module in a.named_modules():
    print(name)
