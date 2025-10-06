__all__ = ['OrbitDict', 'OrbitalElements', 'elem2rv']
import dataclasses
import random
import warnings
from collections import UserList
from typing import Any, Iterable, TypedDict, cast

import torch
from typing_extensions import Self

eps = 1e-15


class OrbitDict(TypedDict):
    id: int
    eccentricity: float
    semi_major_axis: float
    inclination: float
    right_ascension_of_the_ascending_node: float
    argument_of_perigee: float
    true_anomaly: float


@dataclasses.dataclass(frozen=True)
class OrbitalElement:
    """Orbital elements of a satellite.

    Refer to https://en.wikipedia.org/wiki/Orbital_
    """
    semi_major_axis: float
    eccentricity: float
    inclination: float
    right_ascension_of_the_ascending_node: float
    argument_of_perigee: float
    true_anomaly: float

    def to_dict(self) -> OrbitDict:
        d = dataclasses.asdict(self)
        return cast(OrbitDict, d)

    @classmethod
    def from_dict(cls, orbit: OrbitDict) -> Self:
        d = cast(dict[str, Any], orbit.copy())
        return cls(**d)

    @property
    def data(self) -> list[float]:
        data = dataclasses.astuple(self)
        return cast(list[float], data)

    @classmethod
    def sample(cls) -> Self:

        return cls(
            round(random.uniform(6.8e6, 8e6), 1),
            round(random.uniform(0, 0.005), 6),
            round(random.uniform(0, torch.pi), 1),
            round(random.uniform(0, 2 * torch.pi), 1),
            round(random.uniform(0, 2 * torch.pi), 1),
            round(random.uniform(0, 2 * torch.pi), 1),
        )


class OrbitalElements(UserList[OrbitalElement]):

    @classmethod
    def from_dicts(cls, configs: Iterable[OrbitDict]) -> Self:
        return cls([OrbitalElement.from_dict(config) for config in configs])

    def to_dicts(self) -> list[OrbitDict]:
        return [element.to_dict() for element in self]

    def sample(cls, n: int) -> Self:
        return cls([OrbitalElement.sample(i) for i in range(n)])

    def to_tensor(self) -> torch.Tensor:
        data = torch.tensor([element.data for element in self])
        return data


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
    data = elements.to_tensor()
    (
        semi_major_axis,
        eccentricity,
        inclination,
        right_ascension_of_the_ascending_node,
        argument_of_perigee,
        true_anomaly,
    ) = data.unbind(-1)

    if torch.any(1.0 + eccentricity * torch.cos(true_anomaly) < eps):
        warnings.warn(
            'WARNING: Radius is near infinite in elem2rv conversion.')

    # Calculate the semilatus rectum and the radius #
    p = semi_major_axis * (1.0 - eccentricity * eccentricity)
    r = p / (1.0 + eccentricity * torch.cos(true_anomaly))
    theta = argument_of_perigee + true_anomaly
    r1 = r * (
        torch.cos(theta) * torch.cos(right_ascension_of_the_ascending_node) -
        torch.cos(inclination) * torch.sin(theta) *
        torch.sin(right_ascension_of_the_ascending_node))
    r2 = r * (
        torch.cos(theta) * torch.sin(right_ascension_of_the_ascending_node) +
        torch.cos(inclination) * torch.sin(theta) *
        torch.cos(right_ascension_of_the_ascending_node))
    r3 = r * (torch.sin(theta) * torch.sin(inclination))

    if torch.any(torch.abs(p) < eps):
        if torch.any(torch.abs(1.0 - eccentricity) < eps):
            # Rectilinear orbit #
            raise ValueError('elem2rv does not support rectilinear orbits')
        # Parabola #
        rp = -semi_major_axis
        p = 2.0 * rp

    h = torch.sqrt(mu * p)
    v1 = -mu / h * (
        torch.cos(right_ascension_of_the_ascending_node) *
        (eccentricity * torch.sin(argument_of_perigee) + torch.sin(theta)) +
        torch.cos(inclination) *
        (eccentricity * torch.cos(argument_of_perigee) + torch.cos(theta)) *
        torch.sin(right_ascension_of_the_ascending_node))
    v2 = -mu / h * (
        torch.sin(right_ascension_of_the_ascending_node) *
        (eccentricity * torch.sin(argument_of_perigee) + torch.sin(theta)) -
        torch.cos(inclination) *
        (eccentricity * torch.cos(argument_of_perigee) + torch.cos(theta)) *
        torch.cos(right_ascension_of_the_ascending_node))
    v3 = mu / h * (eccentricity * torch.cos(argument_of_perigee) +
                   torch.cos(theta)) * torch.sin(inclination)

    return torch.stack([r1, r2, r3], dim=-1), torch.stack([v1, v2, v3], dim=-1)
