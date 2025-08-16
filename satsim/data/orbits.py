__all__ = ['OrbitDict', 'OrbitalElements', 'elem2rv']
import warnings
import torch
import dataclasses
from typing import Any, TypedDict, cast
from typing_extensions import Self

eps = 1e-15


class OrbitDict(TypedDict):
    eccentricity: Any
    semi_major_axis: Any
    inclination: Any
    right_ascension_of_the_ascending_node: Any
    argument_of_perigee: Any
    true_anomaly: Any


@dataclasses.dataclass(frozen=True)
class OrbitalElements:
    """Orbital elements of a satellite.

    Refer to https://en.wikipedia.org/wiki/Orbital_elements.
    """
    semi_major_axis: torch.Tensor
    eccentricity: torch.Tensor
    inclination: torch.Tensor
    right_ascension_of_the_ascending_node: torch.Tensor
    argument_of_perigee: torch.Tensor
    true_anomaly: torch.Tensor = torch.zeros(1)

    def to_dict(self) -> OrbitDict:
        d: dict[str, torch.Tensor] = dataclasses.asdict(self)
        d = {k: v.tolist() for k, v in d.items()}
        return cast(OrbitDict, d)

    @classmethod
    def from_dict(cls, orbit: OrbitDict) -> Self:
        d = cast(dict[str, Any], orbit.copy())
        d = {k: torch.tensor(v) for k, v in d.items()}
        return cls(**d)

    @property
    def data(self) -> list[torch.Tensor]:
        _, *data = dataclasses.astuple(self)
        return cast(list[torch.Tensor], data)

    @classmethod
    def sample(cls, size: int | list[int]) -> Self:
        if isinstance(size, int):
            size = [size]

        lower_bound = torch.tensor([6.8e6, 0., 0., 0., 0.]).expand(*size, 5)
        upper_bound = torch.tensor([8e6, 0.005, 180, 360,
                                    360]).expand(*size, 5)
        dist = torch.distributions.Uniform(lower_bound, upper_bound)
        sample = dist.sample()
        (
            semi_major_axis,
            eccentricity,
            inclination,
            right_ascension_of_the_ascending_node,
            argument_of_perigee,
        ) = sample.unbind(-1)
        semi_major_axis = torch.round(semi_major_axis, decimals=1)
        eccentricity = torch.round(eccentricity, decimals=6)
        inclination = torch.round(inclination, decimals=1)
        right_ascension_of_the_ascending_node = torch.round(
            right_ascension_of_the_ascending_node,
            decimals=1,
        )
        argument_of_perigee = torch.round(argument_of_perigee, decimals=1)

        return cls(
            semi_major_axis=semi_major_axis,
            eccentricity=eccentricity,
            inclination=inclination,
            right_ascension_of_the_ascending_node=
            right_ascension_of_the_ascending_node,
            argument_of_perigee=argument_of_perigee,
        )


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

    if torch.any(
            1.0 +
            elements.eccentricity * torch.cos(elements.true_anomaly) < eps):
        warnings.warn(
            'WARNING: Radius is near infinite in elem2rv conversion.')

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

    if torch.any(torch.abs(p) < eps):
        if torch.any(torch.abs(1.0 - elements.eccentricity) < eps):
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
