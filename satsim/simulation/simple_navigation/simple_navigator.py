__all__ = [
    'SimpleNavigator',
    'SimpleNavStateDict',
    'AttitudeData',
    'TranslationData',
]

from typing import TypedDict

import torch

from satsim.architecture import Module


class AttitudeData(TypedDict):
    mrp_attitude_in_J2000: torch.Tensor  # [3]  [-]    Current spacecraft attitude (MRPs) of body relative to inertial
    angular_velocity_in_J2000: torch.Tensor  # [3]  [r/s]  Current spacecraft angular velocity vector of body frame B relative to inertial frame N, in B frame components
    sun_position_in_body: torch.Tensor  # [3]  [m]    Current sun pointing vector in body frame


class TranslationData(TypedDict):
    position_in_J2000: torch.Tensor  # [3]  [m]    Current inertial spacecraft position vector in inertial frame N components
    velocity_in_J2000: torch.Tensor  # [3]  [m/s]  Current inertial velocity of the spacecraft in inertial frame N components
    total_accumulated_delta_velocity_in_J2000: torch.Tensor  # [3]  [m/s]  Total accumulated delta-velocity for s/c


class SimpleNavStateDict(TypedDict):
    navigation_errors: torch.Tensor


class SimpleNavigator(Module[SimpleNavStateDict]):

    def __init__(
        self,
        *args,
        noise_process_covariance_matrix: torch.Tensor | None = None,
        noise_walk_bounds: torch.Tensor | None = None,
        cross_correlation_for_translation: bool = False,
        cross_correlation_for_attitude: bool = False,
        **kwargs,
    ):
        """Initialize the SimpleNavigator with noise and correlation parameters.

        Args:
            noise_process_covariance_matrix (torch.Tensor, optional): Covariance matrix for the noise process.
                If None, a zero matrix of shape [18, 18] is used. Defaults to torch.zeros([18, 18]).
            noise_walk_bounds (torch.Tensor, optional): Bounds for the noise walk process.
                If None, a tensor of shape [18] filled with -1.0 is used. Defaults to None.
            cross_correlation_for_translation (bool, optional): Whether to include cross-correlation for translation.
                Defaults to False.
            cross_correlation_for_attitude (bool, optional): Whether to include cross-correlation for attitude.
                Defaults to False.
        """
        super().__init__(*args, **kwargs)

        noise_process_covariance_matrix = torch.zeros(
            18, 18
        ) if noise_process_covariance_matrix is None else noise_process_covariance_matrix
        noise_walk_bounds = torch.full(
            [18], -1.0) if noise_walk_bounds is None else noise_walk_bounds
        noise_propagate_matrix = torch.eye(18)
        for i, j in zip([0, 1, 2], [3, 4, 5]):
            noise_propagate_matrix[
                i,
                j] = self._timer.dt if cross_correlation_for_translation else 0.
        for i, j in zip([6, 7, 8], [9, 10, 11]):
            noise_propagate_matrix[
                i,
                j] = self._timer.dt if cross_correlation_for_attitude else 0.

        self.register_buffer(
            "noise_process_covariance_matrix",
            noise_process_covariance_matrix,
        )
        self.register_buffer(
            "noise_walk_bounds",
            noise_walk_bounds,
        )
        self.register_buffer(
            "noise_propagate_matrix",
            noise_propagate_matrix,
        )

    def reset(self) -> SimpleNavStateDict:

        navigation_errors: torch.Tensor = torch.zeros(18)

        return {
            "navigation_errors": navigation_errors,
        }

    def get_all_buffer(
            self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noise_process_covariance_matrix = self.get_buffer(
            "noise_process_covariance_matrix")
        noise_walk_bounds = self.get_buffer("noise_walk_bounds")
        noise_propagate_matrix = self.get_buffer("noise_propagate_matrix")
        return (
            noise_process_covariance_matrix,
            noise_walk_bounds,
            noise_propagate_matrix,
        )

    def forward(
        self,
        state_dict: SimpleNavStateDict,
        *args,
        position_in_J2000: torch.Tensor,
        velocity_in_J2000: torch.Tensor,
        mrp_attitude_in_J2000: torch.Tensor,
        angular_velocity_in_J2000: torch.Tensor,
        total_accumulated_delta_velocity_in_J2000: torch.Tensor,
        sun_position_in_J2000: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[SimpleNavStateDict, tuple[AttitudeData, TranslationData]]:
        navigation_errors = state_dict["navigation_errors"]
        (
            noise_process_covariance_matrix,
            noise_walk_bounds,
            noise_propagate_matrix,
        ) = self.get_all_buffer()

        sun_position_in_body = self.compute_sun_position_in_body(
            position_in_J2000,
            mrp_attitude_in_J2000,
            sun_position_in_J2000,
        )
        navigation_errors = self.compute_errors(
            noise_propagate_matrix,
            navigation_errors,
            noise_process_covariance_matrix,
            noise_walk_bounds,
        )

        estimated_attitude_state, estimated_translation_state = self.apply_errors(
            navigation_errors,
            mrp_attitude_in_J2000,
            angular_velocity_in_J2000,
            sun_position_in_body,
            position_in_J2000,
            velocity_in_J2000,
            total_accumulated_delta_velocity_in_J2000,
        )

        state_dict["navigation_errors"] = navigation_errors
        return state_dict, (estimated_attitude_state,
                            estimated_translation_state)

    def compute_sun_position_in_body(
        self,
        position_in_J2000: torch.Tensor,
        mrp_attitude_in_J2000: torch.Tensor,
        sun_position: torch.Tensor | None,
    ) -> torch.Tensor:
        if sun_position:
            sc2SunInrtl = sun_position - position_in_J2000
            sc2SunInrtl = torch.nn.functional.normalize(sc2SunInrtl, dim=0)
            dcm_BN = MRP2C(mrp_attitude_in_J2000)
            return dcm_BN @ sc2SunInrtl
        else:
            return torch.zeros_like(sun_position)

    def apply_errors(
        self,
        navigation_errors: torch.Tensor,
        mrp_attitude_in_J2000,
        angular_velocity_in_J2000,
        sun_position_in_body,
        position_in_J2000,
        velocity_in_J2000,
        total_accumulated_delta_velocity_in_J2000,
    ) -> tuple[AttitudeData, TranslationData]:
        estimated_translation_state = TranslationData(
            position_in_J2000 + navigation_errors[0:3],
            velocity_in_J2000=velocity_in_J2000 + navigation_errors[3:6],
            total_accumulated_delta_velocity_in_J2000=
            total_accumulated_delta_velocity_in_J2000 +
            navigation_errors[15:18],
        )

        dcm_OT = MRP2C(navigation_errors[12:15])
        estimated_attitude_state = AttitudeData(
            mrp_attitude_in_J2000=addMRP(mrp_attitude_in_J2000,
                                         navigation_errors[6:9]),
            angular_velocity_in_J2000=angular_velocity_in_J2000 +
            navigation_errors[9:12],
            sun_position_in_body=dcm_OT @ sun_position_in_body,
        )
        return estimated_attitude_state, estimated_translation_state

    def compute_errors(
        self,
        noise_propagate_matrix: torch.Tensor,
        navigation_errors: torch.Tensor,
        noise_process_covariance_matrix: torch.Tensor,
        noise_walk_bounds: torch.Tensor,
    ) -> torch.Tensor:

        if noise_propagate_matrix.shape != noise_process_covariance_matrix.shape:
            raise ValueError("Matrix size mismatch in Gauss Markov model")
        if noise_walk_bounds.shape[0] != 18:
            raise ValueError(
                "State bounds size mismatch in Gauss Markov model")

        ran_nums = torch.randn(
            18,
            device=noise_process_covariance_matrix.device,
        )
        error_vector = noise_process_covariance_matrix @ ran_nums
        navigation_errors = noise_propagate_matrix @ navigation_errors
        navigation_errors = navigation_errors + error_vector

        mask = noise_walk_bounds > 0
        clipped = torch.where(
            mask & (navigation_errors.abs() > noise_walk_bounds),
            noise_walk_bounds * navigation_errors.sign(),
            navigation_errors,
        )
        navigation_errors = clipped
        return navigation_errors


def addMRP(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    den = 1 + torch.dot(q1, q1) * torch.dot(q2, q2) - 2 * torch.dot(q1, q2)
    if torch.abs(den) < 0.1:
        q2 = -q2 / torch.dot(q2, q2)
        den = 1 + torch.dot(q1, q1) * torch.dot(q2, q2) - 2 * torch.dot(q1, q2)

    num = (1 - torch.dot(q1, q1)) * q2 + \
          (1 - torch.dot(q2, q2)) * q1 + \
          2 * torch.cross(q1, q2)

    q = num / den
    if torch.dot(q, q) > 1.0:
        q = -q / torch.dot(q, q)
    return q


def MRP2C(q: torch.Tensor) -> torch.Tensor:
    q1, q2, q3 = q.unbind()
    q1_sq, q2_sq, q3_sq = q1**2, q2**2, q3**2
    d1 = q1_sq + q2_sq + q3_sq
    S = 1 - d1
    d = (1 + d1)**2

    c00 = 4 * (2 * q1_sq - d1) + S**2
    c01 = 8 * q1 * q2 + 4 * q3 * S
    c02 = 8 * q1 * q3 - 4 * q2 * S
    c10 = 8 * q2 * q1 - 4 * q3 * S
    c11 = 4 * (2 * q2_sq - d1) + S**2
    c12 = 8 * q2 * q3 + 4 * q1 * S
    c20 = 8 * q3 * q1 + 4 * q2 * S
    c21 = 8 * q3 * q2 - 4 * q1 * S
    c22 = 4 * (2 * q3_sq - d1) + S**2

    C = torch.stack([
        torch.stack([c00, c01, c02]),
        torch.stack([c10, c11, c12]),
        torch.stack([c20, c21, c22]),
    ])
    return C / d
