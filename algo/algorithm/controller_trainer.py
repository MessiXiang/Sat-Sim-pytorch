import json
import os
from copy import deepcopy
from typing import Iterable

import torch
from matplotlib import pyplot as plt
from todd.configs import PyConfig
from todd.patches.py_ import json_dump, json_load
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .enviroment.attitude_control import DATA_DIM, AttitudeControlEnviroment
from .model import AttitudeControlMLP
from .utils import InputNormalizer


class ControllerTrainer:

    def __init__(
        self,
        config: PyConfig,
    ):
        self._env = AttitudeControlEnviroment(**config.enviroment)
        self._num_envs: int = config.enviroment.num_envs
        self._episode_num: int = config.train.epoch
        self._episode_length: int = config.train.episode_length
        self._model = AttitudeControlMLP(DATA_DIM, config.model.hidden_dim)
        self._adam = Adam(
            self._model.parameters(),
            lr=config.train.optim.adam.lr,
            betas=config.train.optim.adam.betas,
        )
        self._input_transform = InputNormalizer(DATA_DIM)
        self._gamma = config.train.gamma

        self._tensorboard_dir = os.path.join(
            config.train.log_dir,
            'tensorboard',
        )

        self._log_dir: str = config.train.log_dir
        self._model_save_dir = os.path.join(self._log_dir, 'checkpoints')
        os.makedirs(self._model_save_dir, exist_ok=True)

    def load_checkpoints(self, path: str) -> None:
        env_build_config = json_load(
            os.path.join(path, 'env_build_config.json'))
        simulator_state_dict = torch.load(
            os.path.join(path, 'simulator_state_dict.pth'))
        normalizer_state_dict = torch.load(
            os.path.join(path, 'normalizer_state_dict.pth'))
        model_state_dict = torch.load(
            os.path.join(path, 'model_checkpoints.pth'))

        self._model.load_state_dict(model_state_dict)
        self._input_transform.load_state_dict(normalizer_state_dict)
        self._env.load_state_dict(
            env_build_config['orbits'],
            env_build_config['tasks'],
            env_build_config['constellation'],
            simulator_state_dict,
        )

    def save_checkpoints(self) -> None:
        path = os.path.join(self._model_save_dir,
                            f'checkpoints_{self._checkpoints_save_time}')
        os.makedirs(path, exist_ok=True)

        orbits_config, constellations_config, simulator_state_dict, tasks_config = self._env.state_dict(
        )
        normalizer_state_dict = self._input_transform.state_dict()
        model_state_dict = self._model.state_dict()
        env_build_config = dict(
            orbits=orbits_config,
            constellation=constellations_config,
            tasks=tasks_config,
        )

        json_dump(env_build_config,
                  os.path.join(path, 'env_build_config.json'),
                  indent=4)
        torch.save(simulator_state_dict,
                   os.path.join(path, 'simulator_state_dict.pth'))
        torch.save(normalizer_state_dict,
                   os.path.join(path, 'normalizer_state_dict.pth'))
        torch.save(model_state_dict, os.path.join(path,
                                                  'model_checkpoints.pth'))

        self._checkpoints_save_time += 1

    def train(self, is_generalized: bool = False) -> None:
        self._checkpoints_save_time = 0
        tensorboard = SummaryWriter(self._tensorboard_dir)

        observation = self._env.reset(True)
        for episode_idx in range(self._episode_num):

            runtime_input_transform = deepcopy(self._input_transform)
            self._input_transform.update(observation)
            observation = runtime_input_transform(observation)

            epoch_losses = []
            for episode_step in tqdm(list(range(self._episode_length))):
                actions = self._model(observation)
                observation, loss = self._env.step(actions)
                epoch_losses.append(loss)

                self._input_transform.update(observation)
                observation = runtime_input_transform(observation)

            epoch_losses = torch.stack(
                epoch_losses).sum() / self._episode_length / self._num_envs
            print(
                f"Episode: {episode_idx}/ {self._episode_num} Average Angle_error: {epoch_losses}"
            )
            tensorboard.add_scalar(
                "Train/Loss",
                epoch_losses,
                episode_idx,
            )

            self._adam.zero_grad()
            epoch_losses.backward()
            self._adam.step()
            if episode_idx % 50 == 0:
                self.save_checkpoints()
                self.test()
            observation = self._env.reset(is_generalized)
        tensorboard.close()

    @torch.no_grad()
    def test(self) -> None:
        observation = self._env.reset()
        observation = self._input_transform(observation)
        angle_errors = []
        for episode_step in range(self._episode_length):
            actions = self._model(observation)
            observation, loss = self._env.step(actions)
            observation = self._input_transform(observation)
            angle_errors.append(loss)

        angle_errors = torch.stack(angle_errors, dim=-1).tolist()

        plt.clf()
        for angle_error in angle_errors:
            plt.plot(angle_error)
        plt.xlabel('Timestep')
        plt.ylabel('Angle Error (rad)')
        plt.title('Angle Error over Time')

        os.makedirs(os.path.join(self._log_dir, 'test'), exist_ok=True)
        plt.savefig(
            os.path.join(self._log_dir, 'test',
                         f'angle_error_{self._checkpoints_save_time}.png'))
