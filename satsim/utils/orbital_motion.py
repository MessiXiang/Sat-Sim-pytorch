from dataclasses import dataclass, fields

import torch

eps = 1e-15


@dataclass
class OrbitalElements:
    semi_major_axis: torch.Tensor
    eccentricity: torch.Tensor
    inclination: torch.Tensor
    right_ascension_of_the_ascending_node: torch.Tensor
    argument_of_perigee: torch.Tensor
    true_anomaly: torch.Tensor


def elem2rv(
    mu: float,
    elements: OrbitalElements,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Translates the orbit elements:

    === ========================= =======
    a   semi-major axis           km
    e   eccentricity
    i   inclination               rad
    AN  ascending node            rad
    AP  argument of periapses     rad
    f   true anomaly angle        rad
    === ========================= =======

    to the inertial Cartesian position and velocity vectors.
    The attracting body is specified through the supplied
    gravitational constant mu (units of km^3/s^2).

    :param mu: gravitational parameter
    :param elements: orbital elements
    :return:   rVec, position vector
    :return:   vVec, velocity vector
    """
    for field in fields(elements):
        name = field.name
        value = getattr(elements, name)
        if isinstance(value, torch.Tensor):
            continue
        setattr(elements, name, torch.tensor(value))

    if 1.0 + elements.eccentricity * torch.cos(elements.true_anomaly) < eps:
        print('WARNING: Radius is near infinite in elem2rv conversion.')

    # Calculate the semilatus rectum and the radius #
    p = elements.semi_major_axis * (
        1.0 - elements.eccentricity * elements.eccentricity)
    r = p / (1.0 + elements.eccentricity * torch.cos(elements.true_anomaly))
    theta = elements.argument_of_perigee + elements.true_anomaly
    r1 = r * (torch.cos(theta) *
              torch.cos(elements.right_ascension_of_the_ascending_node) -
              torch.cos(elements.inclination) * torch.sin(theta) *
              torch.sin(elements.right_ascension_of_the_ascending_node))
    r2 = r * (torch.cos(theta) *
              torch.sin(elements.right_ascension_of_the_ascending_node) +
              torch.cos(elements.inclination) * torch.sin(theta) *
              torch.cos(elements.right_ascension_of_the_ascending_node))
    r3 = r * (torch.sin(theta) * torch.sin(elements.inclination))

    if torch.abs(p) < eps:
        if torch.abs(1.0 - elements.eccentricity) < eps:
            # Rectilinear orbit #
            raise ValueError('elem2rv does not support rectilinear orbits')
        # Parabola #
        rp = -elements.semi_major_axis
        p = 2.0 * rp

    h = torch.sqrt(mu * p)
    v1 = -mu / h * (
        torch.cos(elements.right_ascension_of_the_ascending_node) *
        (elements.eccentricity * torch.sin(elements.argument_of_perigee) +
         torch.sin(theta)) + torch.cos(elements.inclination) *
        (elements.eccentricity * torch.cos(elements.argument_of_perigee) +
         torch.cos(theta)) *
        torch.sin(elements.right_ascension_of_the_ascending_node))
    v2 = -mu / h * (
        torch.sin(elements.right_ascension_of_the_ascending_node) *
        (elements.eccentricity * torch.sin(elements.argument_of_perigee) +
         torch.sin(theta)) - torch.cos(elements.inclination) *
        (elements.eccentricity * torch.cos(elements.argument_of_perigee) +
         torch.cos(theta)) *
        torch.cos(elements.right_ascension_of_the_ascending_node))
    v3 = mu / h * (
        elements.eccentricity * torch.cos(elements.argument_of_perigee) +
        torch.cos(theta)) * torch.sin(elements.inclination)

    return torch.stack([r1, r2, r3], dim=-1), torch.stack([v1, v2, v3], dim=-1)
