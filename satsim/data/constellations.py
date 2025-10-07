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
from collections import UserList
from typing import Iterable, TypedDict

import torch
from attr import asdict, dataclass
from typing_extensions import Self


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


ReactionWheelGroup = tuple[ReactionWheel, ReactionWheel, ReactionWheel]


class ReactionWheelGroups(UserList[ReactionWheelGroup]):

    @classmethod
    def from_dicts(cls, configs: Iterable[ReactionWheelConfigGroup]) -> Self:
        return cls(
            [tuple(ReactionWheel(*c) for c in config) for config in configs])

    def to_dicts(self) -> list[ReactionWheelConfigGroup]:
        return [(dataclasses.asdict(rw) for rw in rw_group)
                for rw_group in self]


Direction = tuple[float, float, float]


class SensorConfig(TypedDict):
    half_field_of_view: float
    camera_direction_B_B: Direction
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
            camera_direction_B_B.append(
                config.get('camera_direction_B_B', (0., 0., 1.)))
            power.append(config['power'])

        return cls(
            half_field_of_view,
            camera_direction_B_B,
            power,
        )

    def to_dicts(self) -> list[SensorConfig]:
        return [
            ReactionWheelConfig(
                mech_to_elec_efficiency=v,
                base_power=cd,
                init_speed=p,
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
            panel_normal_B_B.append(config['panel_normal_B_B'])
            panel_area.append(config['panel_area'])
            panel_efficiency.append(config['panel_efficiency'])
        return cls(panel_normal_B_B, panel_area, panel_efficiency)

    def to_dicts(self) -> list[SolarPanelConfig]:
        return [
            SolarPanelConfig(
                panel_normal_B_B=pn,
                panel_area=pa,
                panel_efficiency=pe,
            ) for pn, pa, pe in zip(self.direction, self.area, self.efficiency)
        ]


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
