__all__ = [
    "ReactionWheelConfig",
    "ReactionWheelConfigGroup",
    "ReactionWheel",
    "ReactionWheelGroup",
    "ReactionWheelGroups",
    "Direction",
    "SensorConfig",
    "Sensor",
    "MRPControlConfig",
    "MRPControl",
    "SolarPanelConfig",
    "SolarPanel",
    "BatteryConfig",
    "Battery",
    "InertiaTuple",
    "ConstellationConfig",
    "Constellation",
]

import dataclasses
import math
import random
from collections import UserList
from typing import Iterable, TypedDict

import torch
from typing_extensions import Self

Direction = tuple[float, float, float]


def sample_direction(n: int) -> list[Direction]:
    phi = [random.uniform(0, 1) * torch.pi for _ in range(n)]
    theta = [random.uniform(-0.5, 0.5) * torch.pi for _ in range(n)]
    directions = [(
        math.cos(t) * math.cos(p),
        math.cos(t) * math.sin(p),
        math.sin(t),
    ) for t, p in zip(theta, phi)]
    return directions


class ReactionWheelConfig(TypedDict):
    efficiency: float
    power: float
    rw_speed_init: float


ReactionWheelConfigGroup = tuple[
    ReactionWheelConfig,
    ReactionWheelConfig,
    ReactionWheelConfig,
]


@dataclasses.dataclass(frozen=True)
class ReactionWheel:
    efficiency: float
    power: float
    rw_speed_init: float

    @classmethod
    def sample(cls) -> Self:
        return cls(
            round(random.uniform(0.5, 0.6), 4),
            round(random.uniform(4., 6.), 3),
            round(random.uniform(-100, 100), 1),
        )


ReactionWheelGroup = tuple[ReactionWheel, ReactionWheel, ReactionWheel]


class ReactionWheelGroups(UserList[ReactionWheelGroup]):

    @classmethod
    def from_dicts(cls, configs: Iterable[ReactionWheelConfigGroup]) -> Self:
        return cls(
            [tuple(ReactionWheel(**c) for c in config) for config in configs])

    @classmethod
    def sample(cls, n: int) -> Self:
        return cls([[ReactionWheel.sample() for _ in range(3)]
                    for _ in range(n)])

    def to_dicts(self) -> list[ReactionWheelConfigGroup]:
        return [[dataclasses.asdict(rw) for rw in rw_group]
                for rw_group in self]


class SensorConfig(TypedDict):
    half_field_of_view: float
    direction: Direction
    power: float


@dataclasses.dataclass(frozen=True)
class Sensor:
    half_field_of_view: list[float]
    direction: list[Direction]
    power: list[float]

    @classmethod
    def from_dicts(cls, configs: Iterable[SensorConfig]) -> Self:
        half_field_of_view = []
        camera_direction_B_B = []
        power = []
        for config in configs:
            half_field_of_view.append(config['half_field_of_view'])
            camera_direction_B_B.append(config.get('direction', (0., 0., 1.)))
            power.append(config['power'])

        return cls(
            half_field_of_view,
            camera_direction_B_B,
            power,
        )

    @classmethod
    def sample(cls, n: int) -> Self:
        return cls(
            [round(random.uniform(0.1, 0.3), 5) for _ in range(n)],
            sample_direction(n),
            [random.uniform(1, 10) for _ in range(n)],
        )

    def to_dicts(self) -> list[SensorConfig]:
        return [
            SensorConfig(
                half_field_of_view=v,
                direction=cd,
                power=p,
            ) for v, cd, p in zip(
                self.half_field_of_view,
                self.direction,
                self.power,
            )
        ]


class MRPControlConfig(TypedDict):
    k: float
    ki: float
    p: float
    integral_limit: float


@dataclasses.dataclass(frozen=True)
class MRPControl:
    k: list[float]
    ki: list[float]
    p: list[float]
    integral_limit: list[float]

    @classmethod
    def from_dicts(cls, configs: Iterable[MRPControlConfig]) -> Self:
        k = []
        ki = []
        p = []
        integral_limit = []
        for config in configs:
            k.append(config['k'])
            ki.append(config['ki'])
            p.append(config['p'])
            integral_limit.append(config['integral_limit'])
        return cls(k, ki, p, integral_limit)

    def to_dicts(self) -> list[MRPControlConfig]:
        return [
            MRPControlConfig(
                k=kk,
                ki=kki,
                p=pp,
                integral_limit=il,
            ) for kk, kki, pp, il in zip(
                self.k,
                self.ki,
                self.p,
                self.integral_limit,
            )
        ]

    @classmethod
    def sample(cls, n: int) -> Self:
        ki = [random.uniform(5e-4, 5e-3) for _ in range(n)]
        return cls(
            [random.uniform(7, 9) for _ in range(n)],
            ki,
            [random.uniform(25, 30) for _ in range(n)],
            [kki / 2 for kki in ki],
        )


class SolarPanelConfig(TypedDict):
    direction: Direction
    area: float
    efficiency: float


@dataclasses.dataclass(frozen=True)
class SolarPanel:
    direction: list[Direction]
    area: list[float]
    efficiency: list[float]

    @classmethod
    def from_dicts(cls, configs: Iterable[SolarPanelConfig]) -> Self:
        panel_normal_B_B = []
        panel_area = []
        panel_efficiency = []
        for config in configs:
            panel_normal_B_B.append(config['direction'])
            panel_area.append(config['area'])
            panel_efficiency.append(config['efficiency'])
        return cls(panel_normal_B_B, panel_area, panel_efficiency)

    def to_dicts(self) -> list[SolarPanelConfig]:
        return [
            SolarPanelConfig(
                direction=pn,
                area=pa,
                efficiency=pe,
            ) for pn, pa, pe in zip(self.direction, self.area, self.efficiency)
        ]

    @classmethod
    def sample(cls, n: int) -> Self:
        return cls(
            sample_direction(n),
            [random.uniform(0.1, 0.5) for _ in range(n)],
            [random.uniform(0.3, 0.42) for _ in range(n)],
        )


class BatteryConfig(TypedDict):
    capacity: float
    percentage: float


@dataclasses.dataclass(frozen=True)
class Battery:
    capacity: list[float]
    percentage: list[float]

    @classmethod
    def from_dicts(cls, configs: Iterable[BatteryConfig]) -> Self:
        capacity = []
        percentage = []
        for config in configs:
            capacity.append(config['capacity'])
            percentage.append(config['percentage'])
        return cls(capacity, percentage)

    def to_dicts(self) -> list[BatteryConfig]:
        capacity, percentage = dataclasses.astuple(self)
        return [
            BatteryConfig(
                capacity=cap,
                percentage=perc,
            ) for cap, perc in zip(capacity, percentage)
        ]

    @classmethod
    def sample(cls, n: int) -> Self:
        return cls(
            [random.uniform(8000, 30000) for _ in range(n)],
            [random.uniform(0.5, 1.0) for _ in range(n)],
        )


InertiaTuple = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]


class ConstellationConfig(TypedDict):
    mass: float
    inertia: InertiaTuple
    reaction_wheels: ReactionWheelConfigGroup
    sensor: SensorConfig
    mrp_control: MRPControlConfig
    solar_panel: SolarPanelConfig
    battery: BatteryConfig


@dataclasses.dataclass(frozen=True)
class Constellation:
    mass: list[float]
    inertia: list[InertiaTuple]
    reaction_wheels: ReactionWheelGroups
    sensor: Sensor
    mrp_control: MRPControl
    solar_panel: SolarPanel
    battery: Battery

    @property
    def num_satellite(self) -> int:
        return len(self.mass)

    def to_dict(self) -> list[ConstellationConfig]:
        reaction_wheels = self.reaction_wheels.to_dicts()
        sensor_config = self.sensor.to_dicts()
        mrp_config = self.mrp_control.to_dicts()
        solar_panel_config = self.solar_panel.to_dicts()
        battery = self.battery.to_dicts()

        return [
            ConstellationConfig(
                mass=m,
                inertia=inertia,
                reaction_wheels=rw,
                sensor=sensor,
                mrp_control=mrp,
                solar_panel=panel,
                battery=b,
            ) for m, inertia, rw, sensor, mrp, panel, b in zip(
                self.mass,
                self.inertia,
                reaction_wheels,
                sensor_config,
                mrp_config,
                solar_panel_config,
                battery,
            )
        ]

    @classmethod
    def from_dicts(
        cls,
        configs: list[ConstellationConfig],
    ) -> Self:
        merged = {key: [d[key] for d in configs] for key in configs[0]}
        reaction_wheels = ReactionWheelGroups.from_dicts(
            merged['reaction_wheels'])
        sensor = Sensor.from_dicts(merged['sensor'])
        mrp_control = MRPControl.from_dicts(merged['mrp_control'])
        solar_panel = SolarPanel.from_dicts(merged['solar_panel'])
        battery = Battery.from_dicts(merged['battery'])
        return cls(
            merged['mass'],
            merged['inertia'],
            reaction_wheels,
            sensor,
            mrp_control,
            solar_panel,
            battery,
        )

    @classmethod
    def sample(cls, n: int) -> Self:
        reaction_wheels = ReactionWheelGroups.sample(n)
        sensor = Sensor.sample(n)
        mrp_control = MRPControl.sample(n)
        solar_panel = SolarPanel.sample(n)
        battery = Battery.sample(n)

        mass = [random.uniform(50, 200) for _ in range(n)]
        inertia = [[random.uniform(50, 200) for _ in range(3)]
                   for _ in range(n)]
        inertia = [(i[0], 0, 0, 0, i[1], 0, 0, 0, i[2]) for i in inertia]
        return cls(
            mass,
            inertia,
            reaction_wheels,
            sensor,
            mrp_control,
            solar_panel,
            battery,
        )
