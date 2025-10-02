__all__ = [
    'ReactionWheel',
    'HoneywellHR12Large',
    'HoneywellHR12Medium',
    'HoneywellHR12Small',
    'HoneywellHR16Large',
    'HoneywellHR16Medium',
    'HoneywellHR16Small',
    'expand',
    'concat',
]
from dataclasses import dataclass, fields
from typing import Iterable, Self

import torch

from satsim.architecture import constants


@dataclass
class ReactionWheel:
    # reaction wheel properties
    mass: float | list[float]
    moment_of_inertia_wrt_spin: float | list[float]
    max_torque: float | list[float]
    max_angular_velocity: float | list[float]
    max_power_efficiency: float | list[float]

    # power properties
    base_power: float | list[float]
    elec_to_mech_efficiency: float | list[float]
    mech_to_elec_efficiency: float | list[float]

    # dynamic params
    angular_velocity_init: float | list[float]

    @classmethod
    def build(
        cls,
        *args,
        mass: float = 1.,
        max_momentum: float,
        max_torque: float = 0.,  # turn off
        max_angular_velocity: float = 0.,  # turn off
        max_power_efficiency: float = -1.,  # turn off
        base_power: float = 0.,
        elec_to_mech_efficiency: float = 1.,
        mech_to_elec_efficiency: float = -1,
        angular_velocity_init: float = 0.,
        **kwargs,
    ) -> Self:

        moment_of_inertia_wrt_spin = max_momentum / max_angular_velocity

        return cls(
            mass=mass,
            moment_of_inertia_wrt_spin=moment_of_inertia_wrt_spin,
            max_torque=max_torque,
            max_angular_velocity=max_angular_velocity,
            max_power_efficiency=max_power_efficiency,
            base_power=base_power,
            elec_to_mech_efficiency=elec_to_mech_efficiency,
            mech_to_elec_efficiency=mech_to_elec_efficiency,
            angular_velocity_init=angular_velocity_init,
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


def concat(reaction_wheels: Iterable[ReactionWheel]) -> ReactionWheel:

    attr_dict = {
        field.name: [
            getattr(reaction_wheel, field.name)
            for reaction_wheel in reaction_wheels
        ]
        for field in fields(ReactionWheel)
    }
    return ReactionWheel(**attr_dict)


class HoneywellHR12Large(ReactionWheel):

    @classmethod
    def build(cls, *args, **kwargs) -> Self:
        return super().build(
            *args,
            mass=9.5,
            max_momentum=50.,
            max_torque=0.2,
            max_angular_velocity=6000. * constants.RPM,
            **kwargs,
        )


class HoneywellHR12Medium(ReactionWheel):

    @classmethod
    def build(cls, *args, **kwargs) -> Self:
        return super().build(
            *args,
            mass=7.0,
            max_momentum=25,
            max_torque=0.2,
            max_angular_velocity=6000. * constants.RPM,
            **kwargs,
        )


class HoneywellHR12Small(ReactionWheel):

    @classmethod
    def build(cls, *args, **kwargs) -> Self:
        return super().build(
            *args,
            mass=6.0,
            max_momentum=12.,
            max_torque=0.2,
            max_angular_velocity=6000. * constants.RPM,
            **kwargs,
        )


class HoneywellHR16Small(ReactionWheel):

    @classmethod
    def build(cls, *args, **kwargs) -> Self:
        return super().build(
            *args,
            mass=9.0,
            max_momentum=50.,
            max_torque=0.2,
            max_angular_velocity=6000. * constants.RPM,
            **kwargs,
        )


class HoneywellHR16Medium(ReactionWheel):

    @classmethod
    def build(cls, *args, **kwargs) -> Self:
        return super().build(
            *args,
            mass=10.4,
            max_momentum=75.,
            max_torque=0.2,
            max_angular_velocity=6000. * constants.RPM,
            **kwargs,
        )


class HoneywellHR16Large(ReactionWheel):

    @classmethod
    def build(cls, *args, **kwargs) -> Self:
        return super().build(
            *args,
            mass=12.,
            max_momentum=100.,
            max_torque=0.2,
            max_angular_velocity=6000. * constants.RPM,
            **kwargs,
        )
