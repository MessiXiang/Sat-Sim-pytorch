import random
from typing import TypedDict
import torch
import tqdm
from copy import deepcopy

from satsim.architecture import Module, constants, VoidStateDict, Timer
from satsim.utils import move_to, dict_recursive_apply
from satsim.simulation.base.battery_base import BatteryStateDict
from satsim.simulation.spacecraft import (
    Spacecraft,
    SpacecraftStateDict,
    SpacecraftStateOutput,
)
from satsim.simulation.gravity import (
    PointMassGravityBody,
    GravityField,
    SpiceInterface,
    Ephemeris,
)
from satsim.simulation.reaction_wheels import (
    HoneywellHR12Small,
    ReactionWheels,
    expand,
)
from satsim.simulation.power import NoBattery, SimpleSolarPanel
from satsim.simulation.eclipse import compute_shadow_factor
from satsim.simulation.simple_navigation import (
    SimpleNavigator, )
from satsim.enviroment.ground_location import (
    GroundLocation,
    GroundLocationStateDict,
    AccessState,
)
from satsim.enviroment.ground_mapping import (
    GroundMapping,
    GroundMappingStateDict,
)
from satsim.fsw_algorithm.location_pointing import (LocationPointing,
                                                    LocationPointingStateDict,
                                                    LocationPointingOutput)
from satsim.fsw_algorithm.mrp_feedback import (
    MRPFeedback,
    MRPFeedbackStateDict,
)
from satsim.fsw_algorithm.reaction_wheel_motor_torque import ReactionWheelMotorTorque
from satsim.data import OrbitalElements, elem2rv


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

    def __init__(self, *args, n: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._n = n

        self._setup_gravity_field()

        reaction_wheels = self._setup_reaction_wheels()

        orbits = OrbitalElements.sample(self._n)
        r, v = elem2rv(constants.MU_EARTH, elements=orbits)

        self._spacecraft = Spacecraft(
            timer=self._timer,
            mass=torch.rand(self._n),
            moment_of_inertia_matrix_wrt_body_point=torch.eye(3).expand(
                self._n, 3, 3),
            position_BNp_N=r,
            velocity_BN_N=v,
            angular_velocity_BN_B=torch.zeros(self._n, 3),
            gravity_field=self._gravity_field,
            reaction_wheels=reaction_wheels,
        )

        self._setup_solar_panel()
        self._battery = NoBattery(timer=self._timer)

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
            camera_direction_B_B=torch.tensor([0, 0, 1]).expand(self._n, 3),
            half_field_of_view=torch.randn(self._n) / 10 + 0.25,
        )

        self._mrp_control = MRPFeedback(
            timer=self._timer,
            k=torch.rand(self._n) * 5 + 5,
            ki=torch.rand(self._n) * 1e-3,
            p=torch.rand(self._n) * 30 + 20,
            integral_limit=torch.rand(self._n) * 1e-3,
        )
        self._reaction_wheel_motor_torque = ReactionWheelMotorTorque(
            timer=self._timer,
            control_axis=torch.eye(3).expand(self._n, 3, 3),
        )

    @property
    def spice_interface(self) -> SpiceInterface:
        return self._gravity_field.spice_interface

    @property
    def reaction_wheels(self) -> ReactionWheels:
        return self._spacecraft.reaction_wheels

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

    def _setup_reaction_wheels(self) -> ReactionWheels:
        reaction_wheels = expand(
            [self._n],
            [
                HoneywellHR12Small.build(
                    mech_to_elec_efficiency=random.random(),
                    base_power=random.random() * 2 + 5,
                ) for _ in range(3)
            ],
        )

        return ReactionWheels(
            timer=self._timer,
            reaction_wheels=reaction_wheels,
        )

    def _setup_solar_panel(self) -> None:
        direction = torch.rand(self._n, 3)
        direction = direction / direction.norm(dim=-1, keepdim=True)

        self._solar_panel = SimpleSolarPanel(
            timer=self._timer,
            panel_direction_in_body=direction,
            panel_area=torch.rand(self._n) / 10 + 0.5,
            panel_efficiency=torch.rand(self._n) / 10 + 0.3,
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
            'attitude']
        angular_velocity_BN_B = spacecraft_state_dict['_hub'][
            'dynamic_params']['angular_velocity']

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

        shadow_factor = compute_shadow_factor(
            r_HN_N=position_SN_N,
            r_PN_N=position_PN_N,
            r_BN_N=spacecraft_state_output.position_BN_N,
            planet_radii=torch.tensor([constants.REQ_EARTH]),
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
            position_in_inertial=spacecraft_state_output.position_BN_N,
            mrp_attitude_in_inertial=attitude_BN,
            sun_position_in_inertial=position_SN_N,
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
        mrp_control_state_dict, (
            attitude_control_torque,
            integral_feedback_output,
        ) = self._mrp_control(
            state_dict=mrp_control_state_dict,
            sigma_BR=pointing_guide_output.attitude_BR,
            omega_BR_B=pointing_guide_output.angular_velocity_BR_B,
            omega_RN_B=torch.zeros_like(
                pointing_guide_output.angular_velocity_BR_B),
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

        reaction_wheels_motor_torque_state_dict = state_dict[
            '_reaction_wheel_motor_torque']
        reaction_wheels_motor_torque_state_dict, (
            motor_torque, ) = self._reaction_wheel_motor_torque(
                state_dict=reaction_wheels_motor_torque_state_dict,
                torque_request_body=attitude_control_torque,  # [3]
                reaction_wheel_spin_axis_in_body=self.reaction_wheels.
                spin_axis_in_body,
            )

        reaction_wheels_state_dict, (
            battery_state_dict, ) = self.reaction_wheels(
                state_dict=reaction_wheels_state_dict,
                battery_state_dict=battery_state_dict,
                motor_torque=motor_torque,
            )
        state_dict['_spacecraft'][
            '_reaction_wheels'] = reaction_wheels_state_dict

        battery_state_dict, _ = self._battery(state_dict=battery_state_dict)
        state_dict['_battery'] = battery_state_dict

        return state_dict, (access_state, )

    def setup_target(
        self,
        state_dict: RemoteSensingConstellationStateDict,
        targets_PCPF: torch.Tensor,
    ) -> RemoteSensingConstellationStateDict:
        pointing_location_state_dict = state_dict['_pointing_location']
        pointing_location_state_dict = self._pointing_location.specify_location_PCPF(
            pointing_location_state_dict,
            targets_PCPF,
            constants.REQ_EARTH,
            constants.REQ_EARTH,
        )
        state_dict['_pointing_location'] = pointing_location_state_dict

        return state_dict


if __name__ == '__main__':
    satellite_number = 1
    timer = Timer(1.)
    simulator = RemoteSensingConstellation(timer=timer, n=satellite_number)
    simulator_state_dict = simulator.reset()
    timer.reset()

    simulator.setup_target(
        simulator_state_dict,
        torch.zeros(satellite_number, 3),
    )

    p_bar = tqdm.tqdm(total=180 / timer.dt + 1)
    access_state: AccessState

    while timer.time <= 180.:
        simulator_state_dictaccess_state, (access_state, ) = simulator(
            state_dict=simulator_state_dict)
        timer.step()
        p_bar.update(1)

        previous_state = deepcopy(simulator_state_dict)

    print(access_state.has_access)
