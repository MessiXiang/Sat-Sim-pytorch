__all__ = ['ReactionWheels', 'create_reaction_wheels']
from dataclasses import dataclass
from typing import Optional

import torch

from satsim.architecture import constants

from .data import ReactionWheelModels, ReactionWheelsOutput
from .reaction_wheel_type_registry import reaction_wheel_type_registry


@dataclass
class ReactionWheels:
    """This class stores all wheel states. All values are of type torch.Tensor with size (...,n)"""
    rWB_B: torch.Tensor  # [3]
    gsHat_B: torch.Tensor  # [3]
    w2Hat0_B: torch.Tensor  # [3]
    w3Hat0_B: torch.Tensor  # [3]
    aOmega: torch.Tensor  # [3]
    bOmega: torch.Tensor  # [3]
    rWcB_B: torch.Tensor  # [3]
    rPrimeWcB_B: torch.Tensor  # [3]
    w2Hat_B: torch.Tensor  # [3]
    w3Hat_B: torch.Tensor  # [3]
    IRWPntWc_B: torch.Tensor  # [3,3]
    IPrimeRWPntWc_B: torch.Tensor  # [3,3]
    rTildeWcB_B: torch.Tensor  # [3,3]
    mass: torch.Tensor
    theta: torch.Tensor
    Omega: torch.Tensor
    Js: torch.Tensor
    Jt: torch.Tensor
    Jg: torch.Tensor
    U_s: torch.Tensor
    U_d: torch.Tensor
    d: torch.Tensor
    J13: torch.Tensor
    current_torque: torch.Tensor
    max_torque: torch.Tensor
    min_torque: torch.Tensor
    friction_coulomb: torch.Tensor
    friction_static: torch.Tensor
    beta_static: torch.Tensor
    cViscous: torch.Tensor
    omegaLimitCycle: torch.Tensor
    friction_torque: torch.Tensor
    omega_before: torch.Tensor
    max_omega: torch.Tensor
    max_power: torch.Tensor
    cOmega: torch.Tensor
    reaction_wheel_model: torch.Tensor  # ReactionWheelModels
    friction_stribeck: torch.Tensor  # bool

    @property
    def num_reaction_wheel(self) -> int:
        return self.reaction_wheel_model.size(
            -1) if self.reaction_wheel_model is not None else 0

    @property
    def is_jitter(self) -> torch.Tensor:
        return self.reaction_wheel_model != ReactionWheelModels.BALANCED_WHEELS

    @property
    def export_key(self) -> list[str]:
        return [
            "rWB_B",
            "gsHat_B",
            "w2Hat0_B",
            "w3Hat0_B",
            "mass",
            "theta",
            "Omega",
            "Js",
            "Jt",
            "Jg",
            "U_s",
            "U_d",
            "d",
            "J13",
            "torque_current",
            "frictionTorque",
            "torque_max",
            "torque_min",
            "fCoulomb",
            "Omega_max",
            "power_max",
            "RWModel",
        ]

    def export(self) -> ReactionWheelsOutput:
        export_dict = {k: getattr(self, k) for k in self.export_key}
        export_dict['linearFrictionRatio'] = torch.zeros_like(
            self.reaction_wheel_model)
        return export_dict

    @classmethod
    def create(
        cls,
        reaction_wheel_type: list[str],
        gsHat_B: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> 'ReactionWheels':
        num_reaction_wheel = len(reaction_wheel_type)
        assert num_reaction_wheel == gsHat_B.size(
            -1), "inconsistency of number reaction wheel"

        reaction_wheel_model = torch.full(
            (1, num_reaction_wheel),
            ReactionWheelModels.BALANCED_WHEELS,
        )
        if 'reaction_wheel_model' in kwargs:
            reaction_wheel_model = kwargs['reaction_wheel_model']
            assert num_reaction_wheel == reaction_wheel_model.size(
                -1), "inconsistency of number reaction wheel"
            assert torch.all(reaction_wheel_model == 1), \
                "Only balanced wheel is supported now."

        use_reaction_wheel_friction = torch.zeros(
            1,
            num_reaction_wheel,
            dtype=torch.bool,
        )
        if 'use_reaction_wheel_friction' in kwargs:
            use_reaction_wheel_friction = kwargs['use_reaction_wheel_friction']
            assert num_reaction_wheel == use_reaction_wheel_friction.size(
                -1), "inconsistency of number reaction wheel"
            assert use_reaction_wheel_friction.dtype is torch.bool, "use_reaction_wheel_friction must be a bool argument."

        use_min_torque = torch.zeros(
            1,
            num_reaction_wheel,
            dtype=torch.bool,
        )
        if 'use_min_torque' in kwargs:
            use_min_torque = kwargs['use_min_torque']
            assert num_reaction_wheel == use_min_torque.size(
                -1), "inconsistency of number reaction wheel"
            assert use_min_torque.dtype is torch.bool, "use_min_torque must be a bool argument."

        use_max_torque = torch.zeros(
            1,
            num_reaction_wheel,
            dtype=torch.bool,
        )
        if 'use_max_torque' in kwargs:
            use_max_torque = kwargs['use_max_torque']
            assert num_reaction_wheel == use_max_torque.size(
                -1), "inconsistency of number reaction wheel"
            assert use_max_torque.dtype is torch.bool, "use_max_torque must be a bool argument."

        max_momentum = torch.zeros(1, num_reaction_wheel)
        if 'max_momentum' in kwargs:
            max_momentum = kwargs['max_momentum']
            assert num_reaction_wheel == max_momentum.size(
                -1), "inconsistency of number reaction wheel"
            assert max_momentum.dtype is torch.float, "max_momentum must be a float argument."

        max_power = torch.full((1, num_reaction_wheel), -1.)
        if 'power_max' in kwargs:
            max_power = kwargs['power_max']
            assert num_reaction_wheel == max_power.size(
                -1), "inconsistency of number reaction wheel"
            assert max_power.dtype is torch.float, "power_max must be a float argument."

        beta_static = torch.full((1, num_reaction_wheel), -1.)
        if 'beta_static' in kwargs:
            beta_static = kwargs['beta_static']
            assert num_reaction_wheel == beta_static.size(
                -1), "inconsistency of number reaction wheel"
            assert beta_static.dtype is torch.float, "beta_static must be a float argument."
            assert torch.any(
                beta_static == 0.
            ), "beta_static cannot be set to zero.  Positive turns it on, negative turns it off"

        (
            max_omega,
            max_torque,
            min_torque,
            friction_coulomb,
            mass,
            u_s,
            u_d,
        ) = _get_type_specific_params(reaction_wheel_type, max_momentum)

        if 'friction_coulomb' in kwargs:
            friction_coulomb = kwargs['friction_coulomb']
            assert num_reaction_wheel == friction_coulomb.size(
                -1), "inconsistency of number reaction wheel"
            assert friction_coulomb.dtype is torch.float, "friction_coulomb must be a float argument."

        friction_static = torch.zeros(1, num_reaction_wheel)
        if 'friction_static' in kwargs:
            friction_static = kwargs['friction_static']
            assert num_reaction_wheel == friction_static.size(
                -1), "inconsistency of number reaction wheel"
            assert friction_static.dtype is torch.float, "friction_static must be a float argument."

        cViscous = torch.zeros(1, num_reaction_wheel)
        if 'cViscous' in kwargs:
            cViscous = kwargs['cViscous']
            assert num_reaction_wheel == cViscous.size(
                -1), "inconsistency of number reaction wheel"
            assert cViscous.dtype is torch.float, "cViscous must be a float argument."

        if 'min_torque' in kwargs:
            min_torque = kwargs['min_torque']
            assert num_reaction_wheel == min_torque.size(
                -1), "inconsistency of number reaction wheel"
            assert min_torque.dtype is torch.float, "min_torque must be a float argument."
        assert not torch.any(min_torque <= 0. and use_min_torque)

        if 'max_torque' in kwargs:
            max_torque = kwargs['max_torque']
            assert num_reaction_wheel == max_torque.size(
                -1), "inconsistency of number reaction wheel"
            assert max_torque.dtype is torch.float, "min_torque must be a float argument."
        assert not torch.any(max_torque <= 0. and use_max_torque)

        if 'max_omega' in kwargs:
            max_omega = kwargs['max_omega']
            assert num_reaction_wheel == max_omega.size(
                -1), "inconsistency of number reaction wheel"
            assert max_omega.dtype is torch.float, "min_torque must be a float argument."

            max_omega = max_omega * constants.RPM

        Js = torch.full((1, num_reaction_wheel), -1)
        Jt = torch.zeros_like(Js)
        Jg = torch.zeros_like(Js)
        Js_direct_set_mask = torch.zeros_like(Js, dtype=torch.bool)
        if 'Js' in kwargs:
            Js = kwargs['Js']
            assert num_reaction_wheel == Js.size(
                -1), "inconsistency of number reaction wheel"
            assert Js.dtype is torch.float, "Js must be a float argument."

            Js_direct_set_mask = Js > 0.
        Js_indirect_set_mask = max_omega > 0. and max_momentum > 0.
        assert torch.all(
            Js_direct_set_mask ^ Js_indirect_set_mask
        ), "Js must be set, either through direct value set or indirectly using max_momentum and max_omega"

        Js = torch.where(Js_direct_set_mask, Js, max_momentum / max_omega)
        Jt = 0.5 * Js
        Jg = Jt

        gsHat_B, w2Hat0_B, w3Hat0_B = _set_hs_hat(gs_hat_b=gsHat_B)

        rWB_B = torch.zeros(3, num_reaction_wheel)
        if 'rWB_B' in kwargs:
            rWB_B = kwargs['rWB_B']
            assert num_reaction_wheel == rWB_B.size(
                -1), "inconsistency of number reaction wheel"
            assert rWB_B.dim() == 2 and rWB_B.size(
                0) == 3, "must be of size [3,n]"
            assert rWB_B.dtype is torch.float, "rWB_B must be a float argument."

        omega = torch.zeros(1, num_reaction_wheel)
        if 'Omega' in kwargs:
            omega = kwargs['omega']
            assert num_reaction_wheel == omega.size(
                -1), "inconsistency of number reaction wheel"
            assert omega.dtype is torch.float, "omega must be a float argument."

        omega = omega * constants.RPM
        theta = torch.zeros(1, num_reaction_wheel)

        friction_coulomb[~use_reaction_wheel_friction] = 0.
        max_torque[~use_max_torque] = -1
        min_torque[~use_min_torque] = 0.

        return cls(
            rWB_B=rWB_B,
            gsHat_B=gsHat_B,
            w2Hat0_B=w2Hat0_B,
            aOmega=torch.zeros_like(omega),
            bOmega=torch.zeros_like(omega),
            rWcB_B=torch.zeros_like(rWB_B),
            rPrimeWcB_B=torch.zeros_like(rWB_B),
            w2Hat_B=torch.zeros_like(w2Hat0_B),
            w3Hat_B=torch.zeros_like(w3Hat0_B),
            IRWPntWc_B=torch.zeros(3, 3, num_reaction_wheel),
            IPrimeRWPntWc_B=torch.zeros(3, 3, num_reaction_wheel),
            rTildeWcB_B=torch.zeros(3, 3, num_reaction_wheel),
            mass=mass,
            theta=theta,
            omega=omega,
            Js=Js,
            Jt=Jt,
            Jg=Jg,
            U_s=u_s,
            u_d=u_d,
            d=torch.zeros_like(Js),
            J13=torch.zeros_like(Js),
            current_torque=torch.zeros_like(max_torque),
            max_torque=max_torque,
            min_torque=min_torque,
            friction_coulomb=friction_coulomb,
            friction_static=friction_static,
            beta_static=beta_static,
            cViscous=cViscous,
            omegaLimitCycle=torch.zeros_like(omega),
            friction_torque=torch.zeros(friction_static),
            omega_before=torch.zeros_like(omega),
            max_omega=max_omega,
            max_power=max_power,
            cOmega=torch.zeros_like(omega),
            reaction_wheel_model=reaction_wheel_model,
            friction_stribeck=torch.zeros_like(
                friction_static,
                dtype=torch.bool,
            ),
        )


def _get_type_specific_params(
    reaction_wheel_type: list[str],
    max_momentum: torch.Tensor,
) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
]:
    type_specific_params_func = [
        reaction_wheel_type_registry.get_type(type_name)
        for type_name in reaction_wheel_type
    ]
    type_specific_params = [
        func(max_momentum[..., i])
        for i, func in enumerate(type_specific_params_func)
    ]

    return tuple(
        torch.cat([tensors[i] for tensors in type_specific_params], dim=-1)
        for i in range(len(type_specific_params[0])))


def _set_hs_hat(
    gs_hat_b: torch.Tensor, ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
    norms = torch.norm(gs_hat_b, dim=0, keepdim=True)
    valid_mask = norms > 1e-10

    if not torch.all(valid_mask):
        raise ValueError(
            f"Error: {torch.sum(~valid_mask)} input vectors are zero or near-zero"
        )

    gs_hat_b = gs_hat_b / norms

    x_axis = torch.tensor([1.0, 0.0, 0.0],
                          device=gs_hat_b.device).reshape(-1, 1)
    w2Hat0_B = torch.cross(gs_hat_b, x_axis.expand_as(gs_hat_b), dim=0)
    w2_norms = torch.norm(w2Hat0_B, dim=0, keepdim=True)

    y_axis = torch.tensor([0.0, 1.0, 0.0],
                          device=gs_hat_b.device).reshape(-1, 1)
    alternative_w2 = torch.cross(gs_hat_b, y_axis.expand_as(gs_hat_b), dim=0)
    switch_mask = w2_norms < 0.01

    w2Hat0_B = torch.where(switch_mask, alternative_w2, w2Hat0_B)
    w2_norms = torch.where(switch_mask,
                           torch.norm(alternative_w2, dim=0, keepdim=True),
                           w2_norms)

    w2Hat0_B = w2Hat0_B / w2_norms

    w3Hat0_B = torch.cross(gs_hat_b, w2Hat0_B, dim=0)

    return gs_hat_b, w2Hat0_B, w3Hat0_B
