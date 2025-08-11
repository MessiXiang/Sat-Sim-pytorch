__all__ = ["ServoControllerStateDict", "ServoController"]

from typing import List, Tuple, TypedDict

import torch

from satsim.architecture import Module


class ServoControllerStateDict(TypedDict):
    rate_tracking_error: torch.Tensor


class ServoController(Module[ServoControllerStateDict]):
    """
    In:
    wheel_spin_axes_body_frame [num, 3] [kgm2]  The reaction wheel spin axis matrix in body frame components
    wheel_spin_inertia [36] [-]     The spin axis inertia for RWs
    wheel_num  [-] [int]       The number of reaction wheels available on vehicle
    spacecraft_inertia_body_frame [9] [kg m^2] Spacecraft Inertia
    integral_gain [N*m] [float]  Integration feedback error on rate error
    error_gain [N*m*s] [float]  Rate error feedback gain applied
    integral_limit [N*m] [float]  Integration limit to avoid wind-up issue
    
    omega_BR_B [3] [r/s]  Current body error estimate of B relative to R in B frame components
    omega_RN_B [3] [r/s]  Reference frame rate vector of the of R relative to N in B frame components
    domega_RN_B [3] [r/s2] Reference frame inertial body acceleration of R relative to N in B frame components
    omega_BastR_B [3] [r/s]   Desired body rate relative to R
    omegap_BastR_B [3] [r/s^2] Body-frame derivative of omega_BastR_B
    wheel_availability [num] [-] The current state of the wheel
    wheel_speeds [num] [r/s] The current angular velocities of the reaction wheel wheel
    known_external_torque_body_frame [3] [N*m] known external torque in body frame vector components
    
    Out:
    torque_requested_body_frame [3] [N*m] Control torque requested
    """

    def __init__(
        self,
        *args,
        integral_gain: float | List[float],
        error_gain: float | List[float],
        integral_limit: float | List[float],
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.register_buffer(
            "_integral_gain",
            torch.tensor(integral_gain),
        )
        self.register_buffer(
            "_error_gain",
            torch.tensor(error_gain),
        )
        self.register_buffer(
            "_integral_limit",
            torch.tensor(integral_limit),
        )

    @property
    def integral_gain(self):
        return self.get_buffer("_integral_gain")

    @property
    def error_gain(self):
        return self.get_buffer("_error_gain")

    @property
    def integral_limit(self):
        return self.get_buffer("_integral_limit")

    def reset(self) -> ServoControllerStateDict:
        return ServoControllerStateDict(rate_tracking_error=torch.zeros(0.))

    def forward(
        self,
        state_dict: ServoControllerStateDict,
        *args,
        omega_BR_B: torch.Tensor,
        omega_RN_B: torch.Tensor,
        domega_RN_B: torch.Tensor,
        omega_BastR_B: torch.Tensor,
        omegap_BastR_B: torch.Tensor,
        wheel_speeds: torch.Tensor,
        wheel_spin_axes_body_frame: torch.Tensor,
        wheel_spin_inertia: torch.Tensor,
        spacecraft_inertia_body_frame: torch.Tensor,
        known_external_torque_body_frame: torch.Tensor,
        **kwargs,
    ) -> Tuple[ServoControllerStateDict, Tuple[torch.Tensor]]:
        omega_BN_B = omega_RN_B + omega_BR_B
        omega_BastN_B = omega_BastR_B + omega_RN_B
        omega_BBast_B = omega_BN_B - omega_BastN_B
        rate_tracking_error = state_dict["rate_tracking_error"]

        use_integral_mask = self.integral_gain > 0
        rate_tracking_error = torch.where(
            use_integral_mask.unsqueeze(-1),
            torch.clamp(
                rate_tracking_error + self._timer._dt * omega_BBast_B,
                -self.integral_limit.unsqueeze(-1),
                self.integral_limit.unsqueeze(-1),
            ),
            torch.zeros_like(omega_BBast_B),
        )

        control_torque = self.error_gain.unsqueeze(
            -1) * omega_BBast_B + self.integral_gain.unsqueeze(
                -1) * rate_tracking_error
        temp = torch.matmul(
            spacecraft_inertia_body_frame,
            omega_BN_B.unsqueeze(-1),
        ).squeeze(-1)
        dot = torch.matmul(
            wheel_spin_axes_body_frame,
            omega_BN_B.unsqueeze(-1),
        ).squeeze(-1)
        scaled_result = wheel_spin_inertia * (dot + wheel_speeds)
        temp = temp + torch.matmul(
            scaled_result.unsqueeze(-2),
            wheel_spin_axes_body_frame,
        ).squeeze(-2)
        control_torque = control_torque - torch.cross(
            omega_BastN_B,
            temp,
            dim=-1,
        )

        temp = domega_RN_B - torch.cross(
            omega_BN_B,
            omega_RN_B,
            dim=-1,
        ) + omegap_BastR_B
        control_torque = control_torque - torch.matmul(
            spacecraft_inertia_body_frame,
            temp.unsqueeze(-1),
        ).squeeze(-1) + known_external_torque_body_frame
        control_torque = -control_torque

        state_dict['rate_tracking_error'] = rate_tracking_error
        return state_dict, (control_torque, )
