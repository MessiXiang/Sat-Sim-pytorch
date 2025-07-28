__all__ = ['TestTimer', 'print_dict']
import time

import torch


class TestTimer:

    def __init__(self, name="code block"):
        self.name = name
        self.start_time = None
        self.checkpoint_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.checkpoint_time = self.start_time
        print(f"{self.name} start")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        total_time = time.time() - self.start_time
        print(f"{self.name} finished. Total time: {total_time:.4f} second")

    def checkpoint(self, name="checkpoint"):
        current_time = time.time()
        elapsed_since_start = current_time - self.start_time
        elapsed_since_last = current_time - self.checkpoint_time
        self.checkpoint_time = current_time
        print(
            f"{name} - since start: {elapsed_since_start:.4f} second | since last checkpoint: {elapsed_since_last:.4f} second"
        )


def print_dict(d: dict, prefix: str = '') -> None:
    for k, v in d.items():

        if isinstance(v, torch.Tensor):
            print(f"{prefix} {k} {v.shape}")
        else:
            print_dict(v, f"{prefix} {k}" + '\t')
