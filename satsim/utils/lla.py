__all__ = [
    "LLA2PCPF",
    "DCM_PCPF2SEZ",
    "PCPF2LLA",
]

import torch
from torch import Tensor


def LLA2PCPF(
    latitude: Tensor,
    longitude: Tensor,
    altitude: Tensor,
    equatorial_radius: float,
    polar_radius: float,
) -> Tensor:
    """
    Convert from latitude, longitude, altitude to planet-centered, planet-fixed coordinates.
    
    Args:
        equatorial_radius: Equatorial radius of the planet
        polar_radius: Polar radius of the planet (None for spherical planet)
        
    Returns:
        Position vector in planet-centered, planet-fixed coordinates
    """
    planet_eccentricity = 0.
    # For spherical planet (polar_radius is None or < 0)
    if polar_radius is not None and polar_radius >= 0:
        planet_eccentricity = 1.0 - polar_radius * polar_radius / (
            equatorial_radius * equatorial_radius)

    sin_phi = torch.sin(latitude)
    n_val = equatorial_radius / torch.sqrt(1.0 - planet_eccentricity *
                                           sin_phi * sin_phi)

    x = (n_val + altitude) * torch.cos(latitude) * torch.cos(longitude)
    y = (n_val + altitude) * torch.cos(latitude) * torch.sin(longitude)
    z = ((1.0 - planet_eccentricity) * n_val + altitude) * sin_phi

    return torch.stack([x, y, z], dim=-1)


def DCM_PCPF2SEZ(latitude: Tensor, longitude: Tensor) -> Tensor:
    """
    Convert from planet-centered, planet-fixed coordinates to satellite-centered, east-north-up coordinates.
    Based on Basilisk's C_PCPF2SEZ function: Euler2(M_PI_2-lat, m1) * Euler3(longitude, m2)
    
    Args:
        latitude: Latitude in radians
        longitude: Longitude in radians
        
    Returns:
        Direction cosine matrix from planet-fixed to SEZ frame
    """

    angle1 = torch.pi / 2 - latitude
    angle2 = longitude

    cos_angle1 = torch.cos(angle1)
    sin_angle1 = torch.sin(angle1)
    cos_angle2 = torch.cos(angle2)
    sin_angle2 = torch.sin(angle2)

    batch_size = latitude.shape[0]

    rot2 = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(angle1)
    rot2[:, 0, 0] = cos_angle1
    rot2[:, 0, 2] = -sin_angle1
    rot2[:, 2, 0] = sin_angle1
    rot2[:, 2, 2] = cos_angle1

    rot3 = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(angle1)
    rot3[:, 0, 0] = cos_angle2
    rot3[:, 0, 1] = sin_angle2
    rot3[:, 1, 0] = -sin_angle2
    rot3[:, 1, 1] = cos_angle2

    result = torch.matmul(rot2, rot3)
    return result


def PCPF2LLA(
    position_in_planet: Tensor,
    equatorial_radius: float,
    polar_radius: float,
) -> Tensor:
    """
    Convert from planet-centered, planet-fixed coordinates to latitude, longitude, altitude.
    Supports both spherical and ellipsoidal planets.
    Based on Basilisk's PCPF2LLA implementation.
    
    Args:
        Planet_Center_Planet_Fixed_position: Position in planet-centered, planet-fixed coordinates
        equatorial_radius: Equatorial radius of the planet
        polar_radius: Polar radius of the planet (None for spherical planet)
        
    Returns:
        Tensor containing [latitude, longitude, altitude]
    """
    x, y, z = position_in_planet.unbind(-1)

    # For spherical planet (polar_radius is None or < 0)
    if polar_radius == equatorial_radius:
        altitude = torch.norm(
            position_in_planet, dim=-1
        ) - equatorial_radius  # Altitude is the height above assumed-spherical planet surface
        longitude = torch.atan2(y, x)
        latitude = torch.atan2(z, torch.sqrt(x * x + y * y))
    else:
        # For ellipsoidal planet - following Basilisk's algorithm
        nr_kapp_iter = 10
        good_kappa_acc = 1.0E-13
        planet_eccentricity2 = 1.0 - polar_radius * polar_radius / (
            equatorial_radius * equatorial_radius)
        kappa = 1.0 / (1.0 - planet_eccentricity2)
        p2 = x * x + y * y
        z2 = z * z

        # Iterative solution for kappa
        for i in range(nr_kapp_iter):
            cI = (p2 + (1.0 - planet_eccentricity2) * z2 * kappa * kappa)
            cI = torch.sqrt(
                cI * cI * cI) / (equatorial_radius * planet_eccentricity2)
            kappaNext = 1.0 + (p2 + (1.0 - planet_eccentricity2) * z2 * kappa *
                               kappa * kappa) / (cI - p2)

            if torch.abs(kappaNext - kappa) < good_kappa_acc:
                break
            kappa = kappaNext

        pSingle = torch.sqrt(p2)
        latitude = torch.atan2(kappa * z, pSingle)
        sin_Phi = torch.sin(latitude)
        N_Val = equatorial_radius / torch.sqrt(1.0 - planet_eccentricity2 *
                                               sin_Phi * sin_Phi)
        altitude = pSingle / torch.cos(latitude) - N_Val
        longitude = torch.atan2(y, x)

    return torch.stack([latitude, longitude, altitude], dim=-1)
