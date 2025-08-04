import torch

from satsim.architecture import constants


def compute_shadow_factor(
    r_HN_N: torch.Tensor,
    r_PN_N: torch.Tensor,
    r_BN_N: torch.Tensor,
    planet_radii: torch.Tensor,
) -> torch.Tensor:
    """
    Compute shadow factors for spacecraft using PyTorch tensor operations.
    
    Args:
        r_HN_N: Sun position [1, 3]
        r_PN_N: Planet positions [n_p, 3]
        r_BN_N: Spacecraft positions [n_s, 3]
        planet_radii: Planet equatorial radii [n_p]
    
    Returns:
        shadow_factors: Shadow factors for each spacecraft [n_s]
    """
    # Compute relative vectors
    s_HP_N = r_HN_N - r_PN_N  # [n_p, 3]
    r_HB_N = r_BN_N - r_HN_N  # [n_s, 3]
    s_BP_N = r_BN_N.unsqueeze(1) - r_PN_N.unsqueeze(0)  # [n_s, n_p, 3]

    # Compute norms
    s_HP_N_norm = torch.norm(s_HP_N, dim=-1)  # [n_p]
    r_HB_N_norm = torch.norm(r_HB_N, dim=-1)  # [n_s]
    s_BP_N_norm = torch.norm(s_BP_N, dim=-1)  # [n_s, n_p]

    # Find closest planet for each spacecraft
    eclipse_mask = r_HB_N_norm.unsqueeze(1) >= s_HP_N_norm  # [n_s, n_p]
    s_BP_N_norm = torch.where(
        eclipse_mask, s_BP_N_norm,
        torch.tensor(float('inf'), device=s_BP_N_norm.device))  # [n_s, n_p]
    eclipse_planet_idx = torch.argmin(s_BP_N_norm, dim=1)  # [n_s]
    valid_eclipse = torch.any(eclipse_mask, dim=1)  # [n_s]

    # Gather closest planet data
    s_BP_N = s_BP_N[torch.arange(s_BP_N.shape[0]),
                    eclipse_planet_idx]  # [n_s, 3]
    planet_radii = planet_radii[eclipse_planet_idx]  # [n_s]
    s_HP_N = s_HP_N[eclipse_planet_idx]  # [n_s, 3]
    s_HP_N_norm = s_HP_N_norm[eclipse_planet_idx]  # [n_s]

    # Eclipse calculations
    f_1 = torch.asin(
        (constants.REQ_SUN * 1000 + planet_radii) / s_HP_N_norm)  # [n_s]
    f_2 = torch.asin(
        (constants.REQ_SUN * 1000 - planet_radii) / s_HP_N_norm)  # [n_s]
    s_0 = -(s_BP_N * s_HP_N).sum(dim=-1) / s_HP_N_norm  # [n_s]
    c_1 = s_0 + planet_radii / torch.sin(f_1)  # [n_s]
    c_2 = s_0 - planet_radii / torch.sin(f_2)  # [n_s]
    s = torch.norm(s_BP_N, dim=-1)  # [n_s]
    l = torch.sqrt(torch.clamp(s * s - s_0 * s_0, min=0))  # [n_s]
    l_1 = c_1 * torch.tan(f_1)  # [n_s]
    l_2 = c_2 * torch.tan(f_2)  # [n_s]

    # Compute shadow factors
    shadow_factors = torch.ones_like(s)  # [n_s]

    # Apply shadow computation for valid eclipses
    eclipse_cases = (l.abs()
                     < l_1.abs()) | ((l.abs() < l_2.abs()) & valid_eclipse)
    shadow_factors = torch.where(
        eclipse_cases, compute_percent_shadow(planet_radii, r_HB_N, s_BP_N),
        shadow_factors)

    return shadow_factors


def compute_percent_shadow(
    planet_radii: torch.Tensor,
    r_HB_N: torch.Tensor,
    s_BP_N: torch.Tensor,
) -> torch.Tensor:
    norm_r_HB_N = torch.norm(r_HB_N, dim=-1)  # [n_s]
    norm_s_BP_N = torch.norm(s_BP_N, dim=-1)  # [n_s]
    a = torch.asin(constants.REQ_SUN * 1000 / norm_r_HB_N)  # [n_s]
    b = torch.asin(planet_radii / norm_s_BP_N)  # [n_s]
    c = torch.acos(
        torch.clamp(
            (-s_BP_N * r_HB_N).sum(dim=-1) / (norm_s_BP_N * norm_r_HB_N), -1.0,
            1.0))  # [n_s]

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
