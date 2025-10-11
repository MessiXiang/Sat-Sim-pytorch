import torch

from .enviroment.enviroment import Enviroment
from .model import Actor, Critic


class Baseline:

    def __init__(self, env: Enviroment, actor: Actor, critic: Critic):
        self._env = env
        self._actor = actor
        self._critic = critic

        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), )
