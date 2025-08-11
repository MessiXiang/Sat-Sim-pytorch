__all__ = [
    'Inertia',
    'SolarPanelDict',
    'SolarPanel',
    'SensorType',
    'SensorDict',
    'Sensor',
    'BatteryDict',
    'Battery',
    'ReactionWheelDict',
    'ReactionWheelDicts',
    'ReactionWheel',
    'ReactionWheels',
    'MRPControlDict',
    'MRPControl',
    'SatelliteDict',
    'Satellite',
]

import dataclasses
import math
import random
from enum import IntEnum, auto
from typing import Any, TypedDict, cast
from typing_extensions import Self

import torch

from ..simulation.reaction_wheels.reaction_wheels import SpinAxis

from .orbits import OrbitalElements

# TODO: rename properties

# yapf: disable
Inertia = tuple[  # kg/m^2
    float, float, float,
    float, float, float,
    float, float, float,
]
# yapf: enable


class SolarPanelDict(TypedDict):
    direction: tuple[float, float, float]
    area: float
    efficiency: float


@dataclasses.dataclass(frozen=True)
class SolarPanel:
    # unit normal vector in the body frame
    direction: tuple[float, float, float]
    area: float  # square meters
    efficiency: float  # [0., 1.]

    def to_dict(self) -> SolarPanelDict:
        solar_panel = dataclasses.asdict(self)
        return cast(SolarPanelDict, solar_panel)

    @property
    def data(self) -> list[float]:
        direction, *data = dataclasses.astuple(self)
        return [*direction, *data]

    @classmethod
    def sample(cls) -> Self:
        phi = random.uniform(0, 1) * math.pi
        theta = random.uniform(-0.5, 0.5) * math.pi
        return cls(
            (
                math.cos(theta) * math.cos(phi),
                math.cos(theta) * math.sin(phi),
                math.sin(theta),
            ),
            random.uniform(0.1, 0.5),
            random.uniform(0.1, 0.5),
        )


class SensorType(IntEnum):
    VISIBLE = auto()
    NEAR_INFRARED = auto()


class SensorDict(TypedDict):
    type: SensorType
    enabled: bool
    half_field_of_view: float
    power: float


@dataclasses.dataclass(frozen=True)
class Sensor:
    type_: SensorType
    enabled: bool
    half_field_of_view: float  # degrees
    power: float  # watts

    def to_dict(self) -> SensorDict:
        d = dataclasses.asdict(self)
        d['type'] = d.pop('type_')
        return cast(SensorDict, d)

    @classmethod
    def from_dict(cls, sensor: SensorDict) -> Self:
        d = cast(dict[str, Any], sensor.copy())
        d['type_'] = SensorType(d.pop('type'))
        return cls(**d)

    @property
    def data(self) -> list[float]:
        return [self.half_field_of_view, self.power]

    @classmethod
    def sample(cls) -> Self:
        return cls(
            SensorType.VISIBLE,
            False,
            random.uniform(0.1, 0.5),
            random.uniform(1, 10),
        )


class BatteryDict(TypedDict):
    capacity: float
    percentage: float


@dataclasses.dataclass(frozen=True)
class Battery:
    capacity: float  # joules
    percentage: float  # [0., 1.]

    def to_dict(self) -> BatteryDict:
        battery = dataclasses.asdict(self)
        return cast(BatteryDict, battery)

    @property
    def static_data(self) -> list[float]:
        return [self.capacity]

    @property
    def dynamic_data(self) -> list[float]:
        return [self.percentage]

    @classmethod
    def sample(cls) -> Self:
        return cls(
            random.uniform(8000, 30000),
            random.uniform(0.1, 1.0),
        )


class ReactionWheelDict(TypedDict):
    rw_type: str
    rw_direction: tuple[float, float, float]
    max_momentum: float
    rw_speed_init: float
    power: float
    efficiency: float


ReactionWheelDicts = tuple[
    ReactionWheelDict,
    ReactionWheelDict,
    ReactionWheelDict,
]


@dataclasses.dataclass(frozen=True)
class ReactionWheel:
    rw_type: str
    spin_axis: SpinAxis | list[SpinAxis]  # unit vector
    angular_velocity_init: float | list[float]  # round per minute
    power: float | list[float]  # watts
    efficiency: float | list[float]  # (0., 1.]

    def to_dict(self) -> ReactionWheelDict:
        reaction_wheel = dataclasses.asdict(self)
        return cast(ReactionWheelDict, reaction_wheel)

    @property
    def static_data(self) -> list[float]:
        return [
            *self.spin_axis,
            self.power,
            self.efficiency,
        ]

    @property
    def dynamic_data(self) -> list[float]:
        return [self.angular_velocity_init]


class MRPControlDict(TypedDict):
    k: float
    ki: float
    p: float
    integral_limit: float


@dataclasses.dataclass(frozen=True)
class MRPControl:
    k: float
    ki: float
    p: float
    integral_limit: float

    def to_dict(self) -> MRPControlDict:
        mrp_control = dataclasses.asdict(self)
        return cast(MRPControlDict, mrp_control)

    @property
    def data(self) -> list[float]:
        return list(dataclasses.astuple(self))


class SatelliteDict(TypedDict):
    inertia: Inertia
    mass: float  # kg
    center_of_mass: tuple[float, float, float]  # m
    orbit_id: int
    solar_panel: SolarPanelDict
    sensor: SensorDict
    battery: BatteryDict
    reaction_wheels: ReactionWheelDicts
    mrp_control: MRPControlDict
    true_anomaly: float  # degrees
    mrp_attitude_bn: tuple[float, float, float]


@dataclasses.dataclass(frozen=True)
class Satellite:
    inertia: torch.Tensor
    mass: torch.Tensor  # kg
    orbit: OrbitalElements
    solar_panel: SolarPanel
    sensor: Sensor
    battery: Battery
    reaction_wheels: ReactionWheel
    mrp_control: MRPControl
    mrp_attitude_bn: tuple[float, float, float]

    def to_dict(self) -> SatelliteDict:
        d = dataclasses.asdict(self)
        d['id'] = d.pop('id_')
        d['orbit'] = d.pop('orbit_id')
        d['solar_panel'] = self.solar_panel.to_dict()
        d['sensor'] = self.sensor.to_dict()
        d['battery'] = self.battery.to_dict()
        d['reaction_wheels'] = tuple(
            reaction_wheel.to_dict()
            for reaction_wheel in self.reaction_wheels)
        d['mrp_control'] = self.mrp_control.to_dict()
        return cast(SatelliteDict, d)

    @classmethod
    def from_dict(
        cls,
        satellite: SatelliteDict,
        orbits: dict[int, OrbitalElements],
    ) -> Self:
        d = cast(dict[str, Any], satellite.copy())
        d['id_'] = d.pop('id')
        d['orbit_id'] = d['orbit']
        d['orbit'] = orbits[d['orbit']]
        d['solar_panel'] = SolarPanel(**d['solar_panel'])
        d['sensor'] = Sensor.from_dict(d['sensor'])
        d['battery'] = Battery(**d['battery'])
        d['reaction_wheels'] = tuple(
            ReactionWheel(**reaction_wheel)
            for reaction_wheel in d.pop('reaction_wheels'))
        d['mrp_control'] = MRPControl(**d.pop('mrp_control'))
        return cls(**d)
