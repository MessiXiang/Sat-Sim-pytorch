__all__ = [
    'SpinAxis',
    'ReactionWheel',
    'HoneywellHR12Large',
    'HoneywellHR12Medium',
    'HoneywellHR12Small',
    'expand',
]
from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Iterable, Self

import torch

from satsim.architecture import constants


# [*batchsize, shape, num_reaction_wheel]
class SpinAxis(IntEnum):
    X = 0
    Y = 1
    Z = 2


@dataclass
class ReactionWheel:
    #
    spin_axis_in_body: SpinAxis | list[SpinAxis]  # [3]
    angular_velocity_init: float | list[float]
    moment_of_inertia_wrt_spin: float | list[float]
    max_torque: float | list[float]
    mass: float | list[float]
    max_angular_velocity: float | list[float]
    max_power_efficiency: float | list[float]

    @classmethod
    def build(
        cls,
        *args,
        spin_axis_in_body: SpinAxis,
        max_momentum: float,
        max_power_efficiency: float = -1.,  # turn off
        max_angular_velocity: float = 0.,  # turn off 
        max_torque: float = 0.,  # turn off
        mass: float = 1.,
        angular_velocity_init: float = 0.,
        **kwargs,
    ) -> Self:

        moment_of_inertia_wrt_spin = max_momentum / max_angular_velocity

        return cls(
            spin_axis_in_body=spin_axis_in_body,
            mass=mass,
            angular_velocity_init=angular_velocity_init,
            moment_of_inertia_wrt_spin=moment_of_inertia_wrt_spin,
            max_torque=max_torque,
            max_angular_velocity=max_angular_velocity,
            max_power_efficiency=max_power_efficiency,
        )


def expand(
    size: Iterable[int],
    reaction_wheels: Iterable[ReactionWheel],
) -> list[ReactionWheel]:
    reaction_wheels = list(reaction_wheels)

    for reaction_wheel in reaction_wheels:
        for field in fields(ReactionWheel):
            attr = field.name
            value = getattr(reaction_wheel, attr)
            if isinstance(value, list):
                raise TypeError(f"{attr} is already a list.")
            value = torch.full(list(size), value).tolist()
            setattr(reaction_wheel, attr, value)

    return reaction_wheels


class HoneywellHR12Large(ReactionWheel):

    @classmethod
    def build(cls, *args, **kwargs) -> Self:
        return super().build(
            *args,
            max_momentum=50.,
            mass=9.5,
            max_angular_velocity=6000. * constants.RPM,
            max_torque=0.2,
            **kwargs,
        )


class HoneywellHR12Medium(ReactionWheel):

    @classmethod
    def build(cls, *args, **kwargs) -> Self:
        return super().build(
            *args,
            max_momentum=25,
            mass=7.0,
            max_angular_velocity=6000. * constants.RPM,
            max_torque=0.2,
            **kwargs,
        )


class HoneywellHR12Small(ReactionWheel):

    @classmethod
    def build(cls, *args, **kwargs) -> Self:
        return super().build(
            *args,
            max_momentum=12.,
            mass=6.0,
            max_angular_velocity=6000. * constants.RPM,
            max_torque=0.2,
            **kwargs,
        )
