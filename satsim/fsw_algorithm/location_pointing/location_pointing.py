__all__ = ['LocationPointing', 'LocationPointingStateDict']

import torch
import torch.nn.functional as F
from typing import TypedDict
from satsim.architecture import Module


class LocationPointingStateDict(TypedDict):
    sigma_BR_old: torch.Tensor  # [3] / [batch, ..., 3]
    e_hat_180_B: torch.Tensor  # [3] / [batch, ..., 3]


class LocationPointing(Module[LocationPointingStateDict]):

    def __init__(
        self,
        *args,
        p_hat_B: torch.Tensor | None = None,  # [b, ..., 3]
        parallel_judgement: float = 0.1,
        use_boresight_rate_dampling: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.register_buffer(
            "parallel_judgement",
            parallel_judgement,
            persistent=False,
        )
        self.register_buffer(
            "use_boresight_rate_dampling",
            use_boresight_rate_dampling,
            persistent=False,
        )

        self.p_hat_B_init = torch.norm(p_hat_B.clone(), dim=-1)
        self.e_hat_180_B_init = None
        self.init = True

    def reset(self) -> LocationPointingStateDict:
        parallel_judgement = self.get_buffer("parallel_judgement")
        p_hat_B = self.p_hat_B_init

        p_norm = torch.norm(p_hat_B, dim=-1)
        if torch.any(p_norm < parallel_judgement):
            print(
                f"locationPoint: vector p_hat_B is not setup as a unit vector. Min norm: {p_norm.min().item}"
            )

        v1 = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(0)  # [1, 3]
        e_hat_180_B = torch.cross(p_hat_B, v1, dim=-1)  # [b, ..., 3]

        e_norm = torch.norm(e_hat_180_B, dim=-1)
        mask = e_norm < parallel_judgement
        if torch.any(mask):
            v1 = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)  # [1, 3]
            e_hat_180_B = torch.where(
                mask.unsqueeze(-1),  # [b, ..., 1]
                torch.cross(p_hat_B, v1, dim=-1),
                e_hat_180_B,
            )

        e_hat_180_B = torch.norm(e_hat_180_B, dim=-1)
        self.e_hat_180_B_init = e_hat_180_B

        batch_shape = p_hat_B.shape[:-1]
        sigma_BR_old = torch.zeros(*batch_shape, 3)

        return {
            "sigma_BR_old": sigma_BR_old,
            "e_hat_180_B": e_hat_180_B,
        }

    def forward(
        self,
        state_dict: LocationPointingStateDict,
        *args,
        r_TN_N: torch.Tensor,  # [m] inertial target location [b, ..., 3]
        r_BN_N: torch.
        Tensor,  # [m] Current inertial spacecraft position vector in inertial frame N components [b, ..., 3]
        sigma_BN: torch.
        Tensor,  # [] Current spacecraft attitude (MRPs) of body relative to inertial N [b, ..., 3]
        sigma_BR: torch.
        Tensor,  # [] Current attitude error estimate (MRPs) of B relative to R [b, ..., 3]
        omega_BN_B: torch.
        Tensor,  # [r/s] Current spacecraft angular velocity vector of body frame B relative to inertial frame N, in B frame components [b, ..., 3]
        omega_BR_B: torch.
        Tensor,  # [r/s] Current body error estimate of B relateive to R in B frame compoonents [b, ..., 3]
        **kwargs,
    ) -> tuple[LocationPointingStateDict, tuple[dict, dict]]:
        parallel_judgement = self.get_buffer("parallel_judgement")
        use_boresight_rate_damping = self.get_buffer(
            "use_boresight_rate_damping")

        sigma_BR_old = state_dict["sigma_BR_old"]  # [b, ..., 3]
        e_hat_180_B = state_dict["e_hat_180_B"]  # [b, ..., 3]

        # calculate r_LS_N
        r_LS_N = r_TN_N - r_BN_N  # [b, ..., 3]

        # principle rotation angle to point pHat at location
        dcm_BN = _mrp_to_dcm(sigma_BN)  # [b, ..., 3, 3]
        r_LS_B = torch.matmul(dcm_BN, r_LS_N.unsqueeze(-1)).squeeze(-1)
        r_hat_LS_B = torch.norm(r_LS_B, dim=-1)  # [b, ..., 3]

        p_hat_B = self.p_hat_B_init  # [b, ..., 3]
        dum1 = torch.sum(p_hat_B * r_hat_LS_B, dim=-1)
        dum1 = torch.clamp(dum1, -1.0, 1.0)
        phi = torch.acos(dum1)

        # calculate sigma_BR
        sigma_BR = torch.zeros_like(r_hat_LS_B)  # [b, ..., 3]
        parallel_mask = phi < parallel_judgement
        sigma_BR = torch.where(
            parallel_mask.unsqueeze(-1),
            torch.zeros_like(sigma_BR),
            sigma_BR,
        )

        non_parallel_mask = ~parallel_mask  # [b, ...]
        if torch.any(non_parallel_mask):
            near_180_mask = (torch.pi - phi) < parallel_judgement
            e_hat_B = torch.where(
                near_180_mask.unsqueeze(-1),  # [b, ..., 1]
                e_hat_180_B.unsqueeze(0).expand_as(sigma_BR),
                torch.cross(p_hat_B, r_hat_LS_B, dim=-1))
            e_hat_B = F.normalize(e_hat_B, dim=-1)
            sigma_BR = torch.where(
                non_parallel_mask.unsqueeze(-1),  # [b, ..., 1]
                -torch.tan(phi.unsqueeze(-1) / 4) * e_hat_B,
                sigma_BR,
            )

        # compute sigma_RN
        sigma_RB = -sigma_BR
        sigma_RN = _add_mrp(sigma_BN, sigma_RB)

        if self.init:
            dt = self._timer.dt
            difference = sigma_BR - sigma_BR_old
            sigma_dot_BR = difference / dt
            binv = _binv_mrp(sigma_BR)
            sigma_dot_BR = 4 * sigma_dot_BR
            omega_BR_B = torch.matmul(binv,
                                      sigma_dot_BR.unsqueeze(-1)).squeeze(-1)
        else:
            self.init = False

        if use_boresight_rate_damping:
            bore_rate_scale = torch.sum(omega_BN_B * r_hat_LS_B,
                                        dim=-1,
                                        keepdim=True)
            bore_rate_B = r_hat_LS_B * bore_rate_scale
            omega_BR_B = omega_BR_B + bore_rate_B

        omega_RN_B = omega_BN_B - omega_BR_B
        omega_RN_N = torch.matmul(dcm_BN.transpose(-2, -1),
                                  omega_RN_B.unsqueeze(-1)).squeeze(-1)

        updated_state = {
            "sigma_BR_old": sigma_BR.clone(),
            "e_hat_180_B": e_hat_180_B,
        }

        att_guidance_message = {
            "sigma_BR": sigma_BR,
            "omega_BR_B": omega_BR_B,
        }
        att_reference_message = {
            "sigma_RN": sigma_RN,
            "omega_RN_N": omega_RN_N,
        }
        return updated_state, (att_guidance_message, att_reference_message)


def _mrp_to_dcm(q: torch.Tensor) -> torch.Tensor:
    """
    Converts Modified Rodrigues Parameters (MRP) to Direction Cosine Matrix (DCM)
    
    Args:
        q: Tensor with shape (b, ..., 3), representing MRP parameters [q1, q2, q3]
        
    Returns:
        C: Tensor with shape (b, ..., 3, 3), representing the direction cosine matrix
    """
    q1 = q[..., 0]
    q2 = q[..., 1]
    q3 = q[..., 2]
    d1 = torch.sum(q**2, dim=-1)
    s = 1 - d1  # 1 - |q|²
    d = (1 + d1)**2  # (1 + |q|²)²

    c = torch.zeros(*q.shape[-1], 3, 3, device=q.device, dtype=q.dtype)

    c[..., 0, 0] = (4 * (2 * q1**2 - d1) + s**2) / d
    c[..., 0, 1] = (8 * q1 * q2 + 4 * q3 * s) / d
    c[..., 0, 2] = (8 * q1 * q3 - 4 * q2 * s) / d

    c[..., 1, 0] = (8 * q2 * q1 - 4 * q3 * s) / d
    c[..., 1, 1] = (4 * (2 * q2**2 - d1) + s**2) / d
    c[..., 1, 2] = (8 * q2 * q3 + 4 * q1 * s) / d

    c[..., 2, 0] = (8 * q3 * q1 + 4 * q2 * s) / d
    c[..., 2, 1] = (8 * q3 * q2 - 4 * q1 * s) / d
    c[..., 2, 2] = (4 * (2 * q3**2 - d1) + s**2) / d

    return c


def _add_mrp(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Computes the composition of two successive MRP rotations (q1 followed by q2)
    
    Args:
        q1: Tensor with shape (b, ..., 3) representing the first MRP rotation
        q2: Tensor with shape (b, ..., 3) representing the second MRP rotation
        
    Returns:
        result: Tensor with shape (b, ..., 3) representing the composed MRP rotation
    """
    s1 = q1.clone()
    q1_dot = torch.sum(s1**2, dim=-1, keepdim=True)
    q2_dot = torch.sum(q2**2, dim=-1, keepdim=True)
    q1q2_dot = torch.sum(q1 * q2, dim=-1, keepdim=True)
    det = 1 + q1_dot * q2_dot - 2 * q1q2_dot

    det_mask = torch.abs(det) < 0.1
    if torch.any(det_mask):
        mag = q1_dot[det_mask]
        s1[det_mask] = -s1[det_mask] / mag
        det = (1 + q1_dot * q2_dot - 2 * q1q2_dot)

        q1_dot_updated = torch.sum(s1**2, dim=-1, keepdim=True)
        q1q2_dot_updated = torch.sum(s1 * q2, dim=-1, keepdim=True)
        det = torch.where(
            det_mask,
            1 + q1_dot_updated * q2_dot - 2 * q1q2_dot_updated,
            det,
        )

    v1 = torch.cross(s1, q2, dim=-1)
    v2 = 2 * v1
    result = (1 - q2_dot) * s1 + v2
    q1_dot_updated = torch.sum(s1**2, dim=-1, keepdim=True)
    v1 = (1 - q1_dot_updated) * q2
    result = (result + v1) / det

    mag = torch.sum(result**2, dim=-1, keepdim=True)
    mag_mask = mag > 1.0
    result = torch.where(
        mag_mask,
        -result / mag,
        result,
    )

    return result


def _binv_mrp(q: torch.Tensor) -> torch.Tensor:
    s2 = torch.sum(q**2, dim=-1, keepdim=True)

    q0 = q[..., 0:1]
    q1 = q[..., 1:2]
    q2 = q[..., 2:3]

    batch_size = q.shape[:-1]
    binv = torch.zeros(*batch_size, 3, 3, device=q.device, dtype=q.dtype)

    # Compute each element of the matrix using MRP B inverse formula
    binv[..., 0, 0] = (1 - s2 + 2 * q0**2).squeeze(-1)
    binv[..., 0, 1] = (2 * (q0 * q1 + q2)).squeeze(-1)
    binv[..., 0, 2] = (2 * (q0 * q2 - q1)).squeeze(-1)

    binv[..., 1, 0] = (2 * (q1 * q0 - q2)).squeeze(-1)
    binv[..., 1, 1] = (1 - s2 + 2 * q1**2).squeeze(-1)
    binv[..., 1, 2] = (2 * (q1 * q2 + q0)).squeeze(-1)

    binv[..., 2, 0] = (2 * (q2 * q0 + q1)).squeeze(-1)
    binv[..., 2, 1] = (2 * (q2 * q1 - q0)).squeeze(-1)
    binv[..., 2, 2] = (1 - s2 + 2 * q2**2).squeeze(-1)

    scale = 1.0 / ((1 + s2)**2)
    binv = binv * scale.squeeze(-1).unsqueeze(-1).unsqueeze(-1)

    return binv
