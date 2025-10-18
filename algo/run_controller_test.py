import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from todd.configs import PyConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from .algorithm.controller_trainer import ControllerTrainer


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def train_controller(rank, world_size, config):
    setup(rank, world_size)
    trainer = ControllerTrainer(config)
    trainer.to_device(rank)


if __name__ == '__main__':
    torch.manual_seed(42)
    random.seed(42)
    torch.set_default_device(0)
    config = PyConfig.load('algo/configs/test.py')
    trainer = ControllerTrainer(config)
    trainer.load_checkpoints('runs/test_run_3/checkpoints/checkpoints_3')
    trainer._checkpoints_save_time = 0
    trainer.test()
