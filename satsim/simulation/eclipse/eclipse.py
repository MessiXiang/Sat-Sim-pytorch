__all__ = ['compute_shadow_factor']

import torch

from satsim.architecture import constants


def compute_shadow_factor(
    position_SN_N: torch.Tensor,
    position_PN_N: torch.Tensor,
    position_BN_N: torch.Tensor,
    planet_radius: torch.Tensor,
) -> torch.Tensor:
    """
    Compute shadow factors for spacecraft using PyTorch tensor operations.

    Args:
        position_SN_N: Sun position in inertial frame [1, 3]
        position_PN_N: Planet positions in inertial frame [n_p, 3]
        position_BN_N: Spacecraft positions in inertial frame [n_s, 3]
        planet_radius: Planet equatorial radii [n_p]

    Returns:
        shadow_factors: Shadow factors for each spacecraft [n_s]
    """
    # Compute relative vectors
    position_SP_N = position_SN_N - position_PN_N  # [n_p, 3]
    position_SB_N = position_SN_N - position_BN_N  # [n_s, 3]
    position_BP_N = position_BN_N.unsqueeze(1) - position_PN_N.unsqueeze(
        0)  # [n_s, n_p, 3]

    # Compute norms
    position_SP_N_norm = torch.norm(position_SP_N, dim=-1)  # [n_p]
    position_SB_N_norm = torch.norm(position_SB_N, dim=-1)  # [n_s]
    position_BP_N_norm = torch.norm(position_BP_N, dim=-1)  # [n_s, n_p]

    # Find closest planet for each spacecraft
    eclipse_mask = position_SB_N_norm.unsqueeze(
        1) >= position_SP_N_norm  # [n_s, n_p]
    position_BP_N_norm = torch.where(
        eclipse_mask, position_BP_N_norm,
        torch.tensor(float('inf'),
                     device=position_BP_N_norm.device))  # [n_s, n_p]
    eclipse_planet_idx = torch.argmin(position_BP_N_norm, dim=1)  # [n_s]
    valid_eclipse = torch.any(eclipse_mask, dim=1)  # [n_s]

    # Gather closest planet data
    position_BP_N = position_BP_N[torch.arange(position_BP_N.shape[0]),
                                  eclipse_planet_idx]  # [n_s, 3]
    planet_radius = planet_radius[eclipse_planet_idx]  # [n_s]
    position_SP_N = position_SP_N[eclipse_planet_idx]  # [n_s, 3]
    position_SP_N_norm = position_SP_N_norm[eclipse_planet_idx]  # [n_s]

    # Eclipse calculations
    f_1 = torch.asin((constants.REQ_SUN * 1000 + planet_radius) /
                     position_SP_N_norm)  # [n_s]
    s_0 = -(position_BP_N *
            position_SP_N).sum(dim=-1) / position_SP_N_norm  # [n_s]
    c_1 = s_0 + planet_radius / torch.sin(f_1)  # [n_s]
    s = torch.norm(position_BP_N, dim=-1)  # [n_s]
    l = torch.sqrt(torch.clamp(s * s - s_0 * s_0, min=0))  # [n_s]
    l_1 = c_1 * torch.tan(f_1)  # [n_s]

    # Compute shadow factors
    shadow_factors = torch.ones_like(s)  # [n_s]

    # Apply shadow computation for valid eclipses
    eclipse_cases = (l.abs() < l_1.abs()) & valid_eclipse
    shadow_factors = torch.where(
        eclipse_cases,
        compute_percent_shadow(planet_radius, position_SB_N, position_BP_N),
        shadow_factors)

    return shadow_factors


def compute_percent_shadow(
    planet_radius: torch.Tensor,
    position_SB_N: torch.Tensor,
    position_BP_N: torch.Tensor,
) -> torch.Tensor:
    position_SB_N_norm = torch.norm(position_SB_N, dim=-1)  # [n_s]
    position_BP_N_norm = torch.norm(position_BP_N, dim=-1)  # [n_s]
    a = torch.asin(constants.REQ_SUN * 1000 /
                   position_SB_N_norm)  # [n_s] Sun apparent disk radius
    b = torch.asin(planet_radius /
                   position_BP_N_norm)  # [n_s] planet apparent disk radius
    c = torch.acos(
        torch.clamp((-position_BP_N * position_SB_N).sum(dim=-1) /
                    (position_BP_N_norm * position_SB_N_norm), -1.0,
                    1.0))  # [n_s] Angular offset between centers

    shadow_fraction = torch.ones_like(a)  # [n_s]

    # Total eclipse: c < b - a
    total_mask = c < b - a
    shadow_fraction = torch.where(total_mask, torch.zeros_like(a),
                                  shadow_fraction)

    # Partial maximum eclipse: c < a - b
    partial_max_mask = c < a - b
    area_sun = torch.pi * a * a
    area_body = torch.pi * b * b
    area = area_sun - area_body
    shadow_fraction = torch.where(partial_max_mask,
                                  1 - area / (torch.pi * a * a),
                                  shadow_fraction)

    # Partial eclipse: c < a + b
    partial_mask = c < a + b
    x = (c * c + a * a - b * b) / (2 * c)
    y = torch.sqrt(torch.clamp(a * a - x * x, min=0))
    area = a * a * torch.acos(torch.clamp(
        x / a, -1.0, 1.0)) + b * b * torch.acos(
            torch.clamp((c - x) / b, -1.0, 1.0)) - c * y
    shadow_fraction = torch.where(
        partial_mask & ~total_mask & ~partial_max_mask,
        1 - area / (torch.pi * a * a), shadow_fraction)

    return shadow_fraction
