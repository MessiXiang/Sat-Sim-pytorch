import torch
from todd.configs import PyConfig

from .algorithm.controller_trainer import ControllerTrainer

if __name__ == '__main__':
    torch.set_default_device('cuda:1')
    torch.set_default_dtype(torch.float32)
    config = PyConfig.load('algo/configs/attitude_control_config.py')
    trainer = ControllerTrainer(config)
    trainer.train()
