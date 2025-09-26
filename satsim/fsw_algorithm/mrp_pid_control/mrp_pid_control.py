__all__ = ['MRPPIDControlStateDict', 'MRPPIDControl']
from typing import TypedDict

import torch

from satsim.architecture import Module


class MRPPIDControlStateDict(TypedDict):
    integral_sigma: torch.Tensor


class MRPPIDControl(Module[MRPPIDControlStateDict]):

    def __init__(
        self,
        *args,
        k: torch.Tensor,
        ki: torch.Tensor,
        p: torch.Tensor,
        integral_limit: torch.Tensor,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.register_buffer('_k', k, persistent=False)
        self.register_buffer('_ki', ki, persistent=False)
        self.register_buffer('_p', p, persistent=False)
        self.register_buffer(
            '_integral_limit',
            integral_limit,
            persistent=False,
        )

    @property
    def k(self) -> torch.Tensor:
        return self.get_buffer('_k')

    @property
    def ki(self) -> torch.Tensor:
        return self.get_buffer('_ki')

    @property
    def p(self) -> torch.Tensor:
        return self.get_buffer('_p')

    @property
    def integral_limit(self) -> torch.Tensor:
        return self.get_buffer('_integral_limit')

    def reset(self) -> MRPPIDControlStateDict:
        return dict(integral_sigma=torch.zeros(3), )

    def forward(
        self,
        state_dict: MRPPIDControlStateDict,
        *args,
        sigma_BR: torch.Tensor,
        omega_BR_B: torch.Tensor,
        omega_RN_B: torch.Tensor,
        domega_RN_B: torch.Tensor,
        wheel_speeds: torch.Tensor,
        inertia_spacecraft_point_b_in_body: torch.Tensor,
        reaction_wheels_inertia_wrt_spin: torch.Tensor,
        reaction_wheels_spin_axis: torch.Tensor,
        **kwargs,
    ) -> tuple[
            MRPPIDControlStateDict,
            tuple[
                torch.Tensor,
            ],
    ]:
        proportional = -self.k.unsqueeze(-1) * sigma_BR

        integral_sigma = state_dict['integral_sigma']
        integral_sigma = integral_sigma + sigma_BR * self._timer.dt

        integral_limit = self.integral_limit.unsqueeze(-1)
        clamp_mask = (torch.abs(integral_sigma) > integral_limit)
        integral_sigma = torch.where(
            clamp_mask,
            torch.copysign(integral_limit, integral_sigma),
            integral_sigma,
        )
        state_dict['integral_sigma'] = integral_sigma

        integral = -self.ki.unsqueeze(-1) * integral_sigma

        derivative = -self.p.unsqueeze(-1) * omega_BR_B

        omega_BN_B = omega_RN_B + omega_BR_B
        j_omega = torch.einsum(
            '...ij,...j -> ...i',
            inertia_spacecraft_point_b_in_body,
            omega_BN_B,
        )
        coriolis_compensation = torch.cross(omega_BN_B, j_omega)

        accel_compensation = torch.einsum(
            '...ij,...j -> ...i',
            inertia_spacecraft_point_b_in_body,
            domega_RN_B,
        )

        feedforward = coriolis_compensation + accel_compensation

        wheel_momentum = reaction_wheels_inertia_wrt_spin * wheel_speeds
        wheel_reaction = -torch.cross(
            wheel_momentum,
            reaction_wheels_spin_axis,
        ).sum(-1)

        # 4. 总控制力矩
        command_torque = proportional + integral + derivative + feedforward + wheel_reaction

        return state_dict, (command_torque, )
