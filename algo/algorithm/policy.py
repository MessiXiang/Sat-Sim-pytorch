import torch

from .enviroment.enviroment import MAX_SATELLITE_NUM, MAX_TASKS_NUM, Enviroment
from .model import SATELLITE_DIM, TASK_DIM, Actor, Critic
from .utils import InputNormalizer


def padding(v: torch.Tensor, split: Iterable[int],
            pad_len: int) -> torch.Tensor:
    split = tuple(split)
    batch_size = len(split)
    padding = torch.zeros(batch_size, pad_len, v.size(-1))
    split_v = torch.split(v, split)

    for i in range(batch_size):
        padding[i, :split[i], :] = split_v[i]

    return padding


def unpadding(v: torch.Tensor, split: Iterable[int]) -> torch.Tensor:
    split = tuple(split)
    pickup = []
    for i in range(v.size(0)):
        valid = v[i, :split[i]]
        pickup.append(valid)
    return torch.cat(pickup, dim=0)


class Baseline:

    def __init__(self, env: Enviroment, actor: Actor, critic: Critic):
        self._env = env
        self._actor = actor
        self._critic = critic

        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), )
