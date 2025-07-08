__all__ = [
    'ReactionWheelsStateDict',
    'ReactionWheel',
    'HoneywellHR12Large',
    'HoneywellHR12Medium',
    'HoneywellHR12Small',
]
from dataclasses import dataclass, field, fields
from typing import Self, TypedDict

import torch

from satsim.architecture import constants

from .data import ReactionWheelModels


class ReactionWheelsStateDict(TypedDict):
    """This class stores all wheel states. All values are of type torch.Tensor with size (...,n)"""
    rWB_B: torch.Tensor  # [3]
    gsHat_B: torch.Tensor  # [3]
    w2Hat0_B: torch.Tensor  # [3]
    w3Hat0_B: torch.Tensor  # [3]
    omega: torch.Tensor
    Js: torch.Tensor
    Jt: torch.Tensor
    Jg: torch.Tensor
    u_s: torch.Tensor
    u_d: torch.Tensor
    max_torque: torch.Tensor
    min_torque: torch.Tensor
    friction_coulomb: torch.Tensor
    friction_static: torch.Tensor
    beta_static: torch.Tensor
    cViscous: torch.Tensor
    reaction_wheel_model: torch.Tensor  # ReactionWheelModels
    mass: torch.Tensor
    current_torque: torch.Tensor
    omegaLimitCycle: torch.Tensor
    friction_torque: torch.Tensor
    omega_before: torch.Tensor
    max_omega: torch.Tensor
    max_power: torch.Tensor
    friction_stribeck: torch.Tensor


zero = lambda: torch.zeros(1)
v3zero = lambda: torch.zeros(3)
m33zero = lambda: torch.zeros(3, 3)


@dataclass
class ReactionWheel:
    rWB_B: torch.Tensor  # [3]
    gsHat_B: torch.Tensor  # [3]
    w2Hat0_B: torch.Tensor  # [3]
    w3Hat0_B: torch.Tensor  # [3]
    omega: torch.Tensor
    Js: torch.Tensor
    Jt: torch.Tensor
    Jg: torch.Tensor
    u_s: torch.Tensor
    u_d: torch.Tensor
    max_torque: torch.Tensor
    min_torque: torch.Tensor
    friction_coulomb: torch.Tensor
    friction_static: torch.Tensor
    beta_static: torch.Tensor
    cViscous: torch.Tensor
    reaction_wheel_model: torch.Tensor  # ReactionWheelModels
    mass: torch.Tensor
    current_torque: torch.Tensor = field(default_factory=zero, init=False)
    omegaLimitCycle: torch.Tensor = field(
        default_factory=lambda: torch.tensor([0.0001]),
        init=False,
    )
    friction_torque: torch.Tensor = field(default_factory=zero, init=False)
    omega_before: torch.Tensor = field(default_factory=zero, init=False)
    max_omega: torch.Tensor
    max_power: torch.Tensor
    friction_stribeck: torch.Tensor = field(
        default_factory=lambda: torch.tensor([False]),
        init=False,
    )

    @classmethod
    def build(
        cls,
        gsHat_B: torch.Tensor,
        reaction_wheel_model: ReactionWheelModels = ReactionWheelModels.
        BALANCED_WHEELS,
        *args,
        max_momentum: float,
        max_power: float = -1.,  # turn off
        beta_static: float = -1.,  # turn off
        max_omega: float = 0.,  # turn off 
        max_torque: float = 0.,  # turn off
        min_torque: float = 0.,  # turn off
        friction_coulomb: float = 0.,  # turn off
        mass: float = 1.,
        u_s: float = 0.,
        u_d: float = 0.,
        friction_static: float = 0.,
        cViscous: float = 0.,
        Js: float = 0.,
        rWB_B: torch.Tensor | None = None,
        omega: float = 0.,
        **kwargs,
    ) -> Self:
        """
        Creates an instance of the class representing a reaction wheel with specified parameters.

        We allow to directly set max_omega, max_torque, min_torque, friction_coulomb, mass, u_s, u_d, but it's recommanded to use type specification.

    Args:
        cls: The class type to instantiate.
        gsHat_B (torch.Tensor): Unit vector in the body frame defining the spin axis of the reaction wheel.
        reaction_wheel_model (ReactionWheelModels, optional): Model type for the reaction wheel. Defaults to ReactionWheelModels.BALANCED_WHEELS.
        *args: Variable length argument list for additional parameters.
        max_momentum (float): Maximum angular momentum capacity of the reaction wheel (Nms).
        max_power (float, optional): Maximum power consumption of the reaction wheel (W). Defaults to -1 (disabled).
        beta_static (float, optional): Static imbalance factor. Defaults to -1 (disabled).
        max_omega (float, optional): Maximum angular velocity of the reaction wheel (rad/s). Defaults to 0 (disabled).
        max_torque (float, optional): Maximum torque the reaction wheel can produce (Nm). Defaults to 0 (disabled).
        min_torque (float, optional): Minimum torque the reaction wheel can produce (Nm). Defaults to 0 (disabled).
        friction_coulomb (float, optional): Coulomb friction coefficient (Nm). Defaults to 0 (disabled).
        mass (float, optional): Mass of the reaction wheel (kg). Defaults to 1.
        u_s (float, optional): Static friction coefficient. Defaults to 0.
        u_d (float, optional): Dynamic friction coefficient. Defaults to 0.
        friction_static (float, optional): Static friction torque (Nm). Defaults to 0.
        cViscous (float, optional): Viscous friction coefficient (Nms/rad). Defaults to 0.
        Js (float, optional): Moment of inertia of the reaction wheel (kgm^2). Defaults to 0.
        rWB_B (torch.Tensor | None, optional): Position vector from the spacecraft's center of mass to the reaction wheel in the body frame (m). Defaults to None.
        omega (float, optional): Initial angular velocity of the reaction wheel (rad/s). Defaults to 0.

    Returns:
        Self: An instance of the class initialized with the specified parameters.
        """
        if reaction_wheel_model != ReactionWheelModels.BALANCED_WHEELS:
            raise ValueError("Only Balanced Wheel is supported")

        if beta_static == 0.:
            raise ValueError(
                "beta_static cannot be set to zero.  Positive turns it on, negative turns it off"
            )

        if (Js > 0.) == (max_omega > 0. and max_momentum > 0.):
            raise ValueError(
                "Js must be set, either through direct value set or indirectly using max_momentum and max_omega"
            )

        Js = max_momentum / max_omega if Js <= 0. else Js
        Jt = 0.5 * Js
        Jg = Jt

        norms = gsHat_B.norm()
        if gsHat_B.shape != torch.Size([3]):
            raise ValueError("gsHat_B size must be torch.Size([3])")
        if norms <= 1e-10:
            raise ValueError("input gsHat_b are zero or near-zero")
        gsHat_B = gsHat_B / norms

        # Calculate w2Hat0_B and w3Hat0_B
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=gsHat_B.device)
        w2Hat0_B = torch.cross(gsHat_B, x_axis)
        w2_norms = torch.norm(w2Hat0_B)
        if w2_norms < 0.01:
            y_axis = torch.tensor([0.0, 1.0, 0.0], device=gsHat_B.device)
            w2Hat0_B = torch.cross(gsHat_B, y_axis)
            w2_norms = torch.norm(w2Hat0_B)
        w2Hat0_B = w2Hat0_B / w2_norms
        w3Hat0_B = torch.cross(gsHat_B, w2Hat0_B)

        rWB_B = rWB_B or torch.zeros(3)
        if rWB_B.shape != torch.Size([3]):
            raise ValueError('rWB_B size must be torch.Size([3])')

        return cls(
            rWB_B=rWB_B,
            gsHat_B=gsHat_B,
            w2Hat0_B=w2Hat0_B,
            w3Hat0_B=w3Hat0_B,
            mass=torch.tensor([mass]),
            omega=torch.tensor([omega]),
            Js=torch.tensor([Js]),
            Jt=torch.tensor([Jt]),
            Jg=torch.tensor([Jg]),
            u_s=torch.tensor([u_s]),
            u_d=torch.tensor([u_d]),
            max_torque=torch.tensor([max_torque]),
            min_torque=torch.tensor([min_torque]),
            friction_coulomb=torch.tensor([friction_coulomb]),
            friction_static=torch.tensor([friction_static]),
            beta_static=torch.tensor([beta_static]),
            cViscous=torch.tensor([cViscous]),
            max_omega=torch.tensor([max_omega]),
            max_power=torch.tensor([max_power]),
            reaction_wheel_model=torch.tensor([reaction_wheel_model]),
        )

    @staticmethod
    def state_dict(
            reaction_wheels: list['ReactionWheel']) -> ReactionWheelsStateDict:
        if len(reaction_wheels) == 0:
            raise ValueError("Reaction Wheel list cannot be empty")

        state_dict = dict()
        for field in fields(ReactionWheel):
            attr = field.name
            values = [
                getattr(reaction_wheel, attr)
                for reaction_wheel in reaction_wheels
            ]
            value = torch.stack(values, dim=-1)
            state_dict[attr] = value

        return state_dict


class HoneywellHR12Large(ReactionWheel):

    @classmethod
    def build(
        cls,
        *args,
        **kwargs,
    ) -> Self:
        return super().build(
            *args,
            max_momentum=50.,
            mass=9.5,
            u_s=4.4e-6,
            u_d=9.1e-7,
            max_omega=6000. * constants.RPM,
            max_torque=0.2,
            min_torque=0.00001,
            friction_coulomb=0.0005,
            **kwargs,
        )


class HoneywellHR12Medium(ReactionWheel):

    @classmethod
    def build(
        cls,
        *args,
        mass: float = 7.0,
        u_s: float = 2.4e-6,
        u_d: float = 4.6e-7,
        max_omega: float = 6000. * constants.RPM,
        max_torque: float = 0.2,
        min_torque: float = 0.00001,
        friction_coulomb: float = 0.0005,
        **kwargs,
    ) -> Self:
        kwargs.pop('max_momentum')
        return super().build(
            *args,
            max_momentum=25.,
            mass=mass,
            u_s=u_s,
            u_d=u_d,
            max_omega=max_omega,
            max_torque=max_torque,
            min_torque=min_torque,
            friction_coulomb=friction_coulomb,
            **kwargs,
        )


class HoneywellHR12Small(ReactionWheel):

    @classmethod
    def build(
        cls,
        *args,
        mass: float = 6.0,
        u_s: float = 1.5e-6,
        u_d: float = 2.2e-7,
        max_omega: float = 6000. * constants.RPM,
        max_torque: float = 0.2,
        min_torque: float = 0.00001,
        friction_coulomb: float = 0.0005,
        **kwargs,
    ) -> Self:
        kwargs.pop('max_momentum')
        return super().build(
            *args,
            max_momentum=12.,
            mass=mass,
            u_s=u_s,
            u_d=u_d,
            max_omega=max_omega,
            max_torque=max_torque,
            min_torque=min_torque,
            friction_coulomb=friction_coulomb,
            **kwargs,
        )
