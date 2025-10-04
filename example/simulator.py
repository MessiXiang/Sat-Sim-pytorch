from typing import TypedDict, cast

import torch
import torch.nn.functional as F

from satsim.architecture import Module, VoidStateDict, constants
from satsim.attitude_control.mrp_feedback import (MRPFeedback,
                                                  MRPFeedbackStateDict)
from satsim.attitude_guidance.location_pointing import (
    LocationPointing, LocationPointingOutput, LocationPointingStateDict)
from satsim.data import OrbitalElements, OrbitDict, elem2rv
from satsim.enviroment.ground_location import (GroundLocation,
                                               GroundLocationStateDict)
from satsim.enviroment.ground_mapping import (GroundMapping,
                                              GroundMappingStateDict)
from satsim.simulation.base.battery_base import BatteryStateDict
from satsim.simulation.eclipse import compute_shadow_factor
from satsim.simulation.gravity import (Ephemeris, GravityField,
                                       PointMassGravityBody, SpiceInterface)
from satsim.simulation.power import NoBattery, SimpleBattery, SimpleSolarPanel
from satsim.simulation.reaction_wheels import (HoneywellHR12Small,
                                               ReactionWheelMotorTorque,
                                               ReactionWheels, concat)
from satsim.simulation.simple_navigation import SimpleNavigator
from satsim.simulation.spacecraft import (Spacecraft, SpacecraftStateDict,
                                          SpacecraftStateOutput)
from satsim.utils import move_to, mrp_to_rotation_matrix

ReactionWheelSpeed = tuple[float, float, float]


class ReactionWheelConfig(TypedDict):
    mech_to_elec_efficiency: list[float]
    base_power: list[float]
    init_speed: list[ReactionWheelSpeed]


class GroundMappingConfig(TypedDict):
    half_field_of_view: list[float]


class MRPControlConfig(TypedDict):
    k: list[float]
    ki: list[float]
    p: list[float]
    integral_limit: list[float]


class SolarPanelConfig(TypedDict):
    panel_normal_B_B: list[tuple[float, float, float]]
    panel_area: list[float]
    panel_efficiency: list[float]


class BatteryConfig(TypedDict):
    capacity: list[float]
    percentage: list[float]


InertiaTuple = tuple[float, float, float, float, float, float, float, float,
                     float]


class ConstellationConfig(TypedDict):
    mass: list[float]
    inertia: list[InertiaTuple]
    reaction_wheel: ReactionWheelConfig
    orbits: list[OrbitDict]
    ground_mapping: GroundMappingConfig
    mrp_control: MRPControlConfig
    solar_panel: SolarPanelConfig
    battery: BatteryConfig


class RemoteSensingConstellationStateDict(TypedDict):
    _gravity_field: VoidStateDict
    _spacecraft: SpacecraftStateDict
    _solar_panel: VoidStateDict
    _battery: BatteryStateDict
    _navigator: VoidStateDict
    _pointing_location: GroundLocationStateDict
    _pointing_guide: LocationPointingStateDict
    _ground_mapping: GroundMappingStateDict
    _mrp_control: MRPFeedbackStateDict
    _reaction_wheel_motor_torque: VoidStateDict


class RemoteSensingConstellation(Module[RemoteSensingConstellationStateDict]):

    def __init__(
        self,
        *args,
        config: ConstellationConfig,
        use_battery: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._n = len(config['mass'])

        self._setup_gravity_field()

        reaction_wheels = self._setup_reaction_wheels(config['reaction_wheel'])

        orbits = OrbitalElements.from_dicts(config['orbits'])
        r, v = elem2rv(constants.MU_EARTH, elements=orbits)

        inertias = torch.tensor(config['inertia']).view(self._n, 3, 3)

        self._spacecraft = Spacecraft(
            timer=self._timer,
            mass=torch.tensor(config['mass']),
            moment_of_inertia_matrix_wrt_body_point=inertias,
            position_BP_N=r,
            velocity_BP_N=v,
            angular_velocity_BN_B=torch.zeros(self._n, 3),
            gravity_field=self._gravity_field,
            reaction_wheels=reaction_wheels,
        )

        self._solar_panel = SimpleSolarPanel(
            timer=self._timer,
            panel_normal_B_B=torch.tensor(
                config['solar_panel']['panel_normal_B_B']),
            panel_area=torch.tensor(config['solar_panel']['panel_area']),
            panel_efficiency=torch.tensor(
                config['solar_panel']['panel_efficiency']),
        )
        if not use_battery:
            self._battery = NoBattery(timer=self._timer)
        else:
            self._battery = SimpleBattery(
                timer=self._timer,
                storage_capacity=torch.tensor(config['battery']['capacity']),
                stored_charge_percentage_init=torch.tensor(
                    config['battery']['percentage']),
            )

        self._navigator = SimpleNavigator(timer=self._timer)

        self._pointing_location = GroundLocation(
            timer=self._timer,
            minimum_elevation=torch.tensor(0.),
        )
        self._pointing_guide = LocationPointing(
            timer=self._timer,
            pointing_direction_B_B=torch.tensor([0., 0.,
                                                 1.]).expand(self._n, 3))
        self._ground_mapping = GroundMapping(
            timer=self._timer,
            minimum_elevation=torch.zeros(self._n),
            maximum_range=torch.full([self._n], 1e9),
            camera_direction_B_B=torch.tensor([0, 0, 1.]).expand(self._n, 3),
            half_field_of_view=torch.tensor(
                config['ground_mapping']['half_field_of_view']),
        )

        self._mrp_control = MRPFeedback(
            timer=self._timer,
            k=torch.tensor(config['mrp_control']['k']),
            ki=torch.tensor(config['mrp_control']['ki']),
            p=torch.tensor(config['mrp_control']['p']),
            integral_limit=torch.tensor(
                config['mrp_control']['integral_limit']),
        )
        self._reaction_wheel_motor_torque = ReactionWheelMotorTorque(
            timer=self._timer,
            control_axis=torch.eye(3).expand(self._n, 3, 3),
        )

    @property
    def num_satellite(self) -> int:
        return self._n

    @property
    def spice_interface(self) -> SpiceInterface:
        return cast(SpiceInterface, self._gravity_field.spice_interface)

    @property
    def reaction_wheels(self) -> ReactionWheels:
        return cast(ReactionWheels, self._spacecraft.reaction_wheels)

    def _setup_gravity_field(self) -> None:
        sun = PointMassGravityBody.create_sun(timer=self._timer)
        earth = PointMassGravityBody.create_earth(
            timer=self._timer,
            is_central=True,
        )

        spice_interface = SpiceInterface(
            timer=self._timer,
            utc_time_init=constants.UTC_TIME_START,
        )

        self._gravity_field = GravityField(
            timer=self._timer,
            spice_interface=spice_interface,
            gravity_bodies=[sun, earth],
        )

    def _setup_reaction_wheels(
        self,
        config: ReactionWheelConfig,
    ) -> ReactionWheels:
        base_powers = config['base_power']
        efficiencys = config['mech_to_elec_efficiency']
        init_speeds = config['init_speed']

        reaction_wheels_0 = []
        reaction_wheels_1 = []
        reaction_wheels_2 = []

        for power, eff, speed in zip(base_powers, efficiencys, init_speeds):
            reaction_wheels_0.append(
                HoneywellHR12Small.build(
                    mech_to_elec_efficiency=eff,
                    base_power=power,
                    angular_velocity_init=speed[0],
                ))
            reaction_wheels_1.append(
                HoneywellHR12Small.build(
                    mech_to_elec_efficiency=eff,
                    base_power=power,
                    angular_velocity_init=speed[1],
                ))
            reaction_wheels_2.append(
                HoneywellHR12Small.build(
                    mech_to_elec_efficiency=eff,
                    base_power=power,
                    angular_velocity_init=speed[2],
                ))

        reaction_wheel_0 = concat(reaction_wheels_0)
        reaction_wheel_1 = concat(reaction_wheels_1)
        reaction_wheel_2 = concat(reaction_wheels_2)

        return ReactionWheels(
            timer=self._timer,
            reaction_wheels=[
                reaction_wheel_0, reaction_wheel_1, reaction_wheel_2
            ],
        )

    # def _get_initial_attitude(self, position_BN_N: torch.Tensor) -> torch.Tensor:

    def forward(
        self,
        state_dict: RemoteSensingConstellationStateDict,
        *args,
        **kwargs,
    ) -> tuple[RemoteSensingConstellationStateDict, tuple]:
        spacecraft_state_dict = state_dict['_spacecraft']

        spacecraft_state_dict: SpacecraftStateDict
        spacecraft_state_output: SpacecraftStateOutput
        spacecraft_state_dict, spacecraft_state_output = self._spacecraft(
            spacecraft_state_dict)
        state_dict['_spacecraft'] = spacecraft_state_dict
        attitude_BN = spacecraft_state_dict['_hub']['dynamic_params'][
            'attitude_BN']
        angular_velocity_BN_B = spacecraft_state_dict['_hub'][
            'dynamic_params']['angular_velocity_BN_B']

        sun_ephemeris: Ephemeris
        earth_ephemeris: Ephemeris
        _, (sun_ephemeris, ) = self.spice_interface(names=['SUN'])
        _, (earth_ephemeris, ) = self.spice_interface(names=['EARTH'])
        sun_ephemeris = move_to(
            sun_ephemeris,
            angular_velocity_BN_B,
        )
        earth_ephemeris = move_to(
            earth_ephemeris,
            angular_velocity_BN_B,
        )
        position_SN_N = sun_ephemeris['position_CN_N']
        position_PN_N = earth_ephemeris['position_CN_N']
        direction_cosine_matrix_PN = earth_ephemeris[
            'direction_cosine_matrix_CN']

        shadow_factor = compute_shadow_factor(
            position_SN_N=position_SN_N,
            position_PN_N=position_PN_N,
            position_BN_N=spacecraft_state_output.position_BN_N,
            planet_radius=torch.tensor([constants.REQ_EARTH * 1e3]),
        )

        battery_state_dict = state_dict['_battery']
        solar_panel_state_dict = state_dict['_solar_panel']

        battery_state_dict: BatteryStateDict
        solar_panel_state_dict, (_, battery_state_dict) = self._solar_panel(
            solar_panel_state_dict,
            position_BN_N=spacecraft_state_output.position_BN_N,
            position_SN_N=position_SN_N,
            attitude_BN=attitude_BN,
            shadow_factor=shadow_factor,
            battery_state_dict=battery_state_dict,
        )

        navigator_state_dict = state_dict['_navigator']
        navigator_state_dict, (_, ) = self._navigator(
            navigator_state_dict,
            position_BN_N=spacecraft_state_output.position_BN_N,
            attitude_BN=attitude_BN,
            position_SN_N=position_SN_N,
        )

        ground_mapping_state_dict = state_dict['_ground_mapping']
        ground_mapping_state_dict, (
            mapping_access_state,
            position_LN_N,
            position_LP_N,
        ) = self._ground_mapping(
            ephemeris=earth_ephemeris,
            position_BN_N=spacecraft_state_output.position_BN_N,  # [n_sc, 3]
            velocity_BN_N=spacecraft_state_output.velocity_BN_N,
            attitude_BN=attitude_BN,
            position_LP_P=self.
            _position_LP_P,  # [n_p, 3] - p stands for mapping point
            equatorial_radius=constants.REQ_EARTH * 1e3,
            polar_radius=constants.REQ_EARTH * 1e3,
        )

        pointing_location_state_dict = state_dict['_pointing_location']
        pointing_location_state_dict, (
            access_state, position_LP_N,
            target_position_LN_N) = self._pointing_location(
                state_dict=pointing_location_state_dict,
                position_BN_N=spacecraft_state_output.position_BN_N,
                velocity_BN_N=spacecraft_state_output.velocity_BN_N,
                ephemeris=earth_ephemeris,
            )
        state_dict['_pointing_location'] = pointing_location_state_dict

        pointing_guide_state_dict = state_dict['_pointing_guide']
        pointing_guide_output: LocationPointingOutput
        pointing_guide_state_dict, pointing_guide_output = self._pointing_guide(
            state_dict=pointing_guide_state_dict,
            position_LN_N=target_position_LN_N,
            position_BN_N=spacecraft_state_output.position_BN_N,
            attitude_BN=attitude_BN,
            angular_velocity_BN_B=angular_velocity_BN_B,
            **kwargs,
        )
        state_dict['_pointing_guide'] = pointing_guide_state_dict

        reaction_wheels_state_dict = state_dict['_spacecraft'][
            '_reaction_wheels']

        mrp_control_state_dict = state_dict['_mrp_control']
        mrp_control_state_dict, attitude_control_torque = self._mrp_control(
            state_dict=mrp_control_state_dict,
            sigma_BR=pointing_guide_output.attitude_BR,
            omega_BR_B=pointing_guide_output.angular_velocity_BR_B,
            omega_RN_B=pointing_guide_output.angular_velocity_RN_B,
            domega_RN_B=torch.zeros_like(
                pointing_guide_output.angular_velocity_BR_B),
            wheel_speeds=reaction_wheels_state_dict['dynamic_params']
            ['angular_velocity'],
            inertia_spacecraft_point_b_in_body=self._spacecraft._hub.
            moment_of_inertia_matrix_wrt_body_point,
            reaction_wheels_inertia_wrt_spin=self.reaction_wheels.
            moment_of_inertia_wrt_spin,
            reaction_wheels_spin_axis=self.reaction_wheels.spin_axis_in_body,
        )
        state_dict['_mrp_control'] = mrp_control_state_dict

        attitude_control_torque = attitude_control_torque[0]
        breakpoint()
        # reaction_wheels_motor_torque_state_dict = state_dict[
        #     '_reaction_wheel_motor_torque']
        # reaction_wheels_motor_torque_state_dict, (
        #     motor_torque, ) = self._reaction_wheel_motor_torque(
        #         state_dict=reaction_wheels_motor_torque_state_dict,
        #         torque_request_body=attitude_control_torque,  # [3]
        #         reaction_wheel_spin_axis_in_body=self.reaction_wheels.
        #         spin_axis_in_body,
        #     )

        reaction_wheels_state_dict, (
            battery_state_dict, ) = self.reaction_wheels(
                state_dict=reaction_wheels_state_dict,
                battery_state_dict=battery_state_dict,
                motor_torque=attitude_control_torque,
            )
        state_dict['_spacecraft'][
            '_reaction_wheels'] = reaction_wheels_state_dict

        battery_state_dict, _ = self._battery(state_dict=battery_state_dict)
        state_dict['_battery'] = battery_state_dict

        position_LB_N = target_position_LN_N - spacecraft_state_output.position_BN_N  # [b, ..., 3]

        # principle rotation angle to point pHat at location
        direction_cosine_matrix_BN = mrp_to_rotation_matrix(
            attitude_BN)  # [b, ..., 3, 3]
        position_LB_B = torch.einsum(
            "...ij,...j->...i",
            direction_cosine_matrix_BN,
            position_LB_N,
        )
        position_LB_B_unit = F.normalize(position_LB_B, dim=-1)

        dum1 = torch.sum(torch.tensor([0., 0., 1.]) * position_LB_B_unit,
                         dim=-1)
        dum1 = torch.clamp(dum1, -1.0, 1.0)
        angle_error = torch.acos(dum1)
        if torch.isnan(angle_error).any():
            breakpoint()
        return state_dict, (
            angle_error,
            mapping_access_state,
            battery_state_dict['stored_charge_percentage'],
        )

    def setup_target(
        self,
        state_dict: RemoteSensingConstellationStateDict,
        targets_PCPF: torch.Tensor,
    ) -> RemoteSensingConstellationStateDict:
        pointing_location_state_dict = state_dict['_pointing_location']
        pointing_location_state_dict = self._pointing_location.specify_location_PCPF(
            pointing_location_state_dict,
            targets_PCPF,
            constants.REQ_EARTH * 1e3,
            constants.REQ_EARTH * 1e3,
        )
        state_dict['_pointing_location'] = pointing_location_state_dict
        self._position_LP_P = targets_PCPF

        return state_dict
