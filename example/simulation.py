import random
from typing import TypedDict
import torch

from satsim.architecture import Module, constants, VoidStateDict
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


class RemoteSensingConstellation(Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n = 50

        self.setup_gravity_field()

        reaction_wheels = self.setup_reaction_wheels()

        orbits = OrbitalElements.sample(self._n)
        r, v = elem2rv(constants.MU_EARTH, elements=orbits)

        self._spacecraft = Spacecraft(
            timer=self._timer,
            mass=torch.rand(self._n),
            moment_of_inertia_matrix_wrt_body_point=torch.eye(3).expand(
                self._n, 3, 3),
            position=r,
            velocity=v,
            gravity_field=self._gravity_field,
            reaction_wheels=reaction_wheels,
        )

        self.setup_solar_panel()
        self._battery = NoBattery(timer=self._timer)

        self._navigator = SimpleNavigator(timer=self._timer)

        self._pointing_location = GroundLocation(
            timer=self._timer,
            minimum_elevation=0.,
        )
        self._pointing_guide = LocationPointing(
            timer=self._timer,
            pointing_direction=torch.tensor([0, 0, 1]).expand(50, 3))
        self._ground_mapping = GroundMapping(
            minimum_elevation=torch.zeros(50),
            maximum_range=torch.full([50], 1e9),
            camera_pos_in_body=torch.zeros(50, 3),
            camera_direction_in_body=torch.tensor([0, 0, 1]).expand(50, 3),
            half_field_of_view=torch.randn(50) / 10 + 0.25,
        )

        self._mrp_control = MRPFeedback(
            k=torch.rand(50) * 5 + 5,
            ki=torch.rand(50) * 1e-3,
            p=torch.rand(50) * 30 + 20,
            integral_limit=torch.rand(50) * 1e-3,
        )
        self._reaction_wheel_motor_torque = ReactionWheelMotorTorque(
            control_axis=torch.eye(3).expand(50, 3, 3), )

    @property
    def spice_interface(self) -> SpiceInterface:
        return self._gravity_field.spice_interface

    @property
    def reaction_wheels(self) -> ReactionWheels:
        return self._spacecraft.reaction_wheels

    def setup_gravity_field(self) -> None:
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

    def setup_reaction_wheels(self) -> ReactionWheels:
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

    def setup_solar_panel(self) -> None:
        direction = torch.rand(self._n, 3)
        direction = direction / direction.norm(dim=-1)

        self._solar_panel = SimpleSolarPanel(
            timer=self._timer,
            direction=direction,
            area=torch.randn(50) / 10 + 0.5,
            efficiency=torch.randn(50) / 10 + 0.3,
        )

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

        sun_ephemeris: Ephemeris
        earth_ephemeris: Ephemeris
        _, (sun_ephemeris, ) = self.spice_interface(names=['SUN'])
        _, (earth_ephemeris, ) = self.spice_interface(names=['EARTH'])
        sun_position = sun_ephemeris['position_in_inertial']
        earth_position = earth_ephemeris['position_in_inertial']

        shadow_factor = compute_shadow_factor(
            r_HN_N=sun_position,
            r_PN_N=earth_position,
            r_BN_N=spacecraft_state_output.position_in_inertial,
            planet_radii=torch.tensor([constants.REQ_EARTH]),
        )

        battery_state_dict = state_dict['_battery']
        solar_panel_state_dict = state_dict['_solar_panel']

        battery_state_dict: BatteryStateDict
        solar_panel_state_dict, (_, battery_state_dict) = self._solar_panel(
            solar_panel_state_dict,
            spacecraft_position_inertial=spacecraft_state_output.
            position_in_inertial,
            sun_position_inertial=sun_position,
            spacecraft_attitude_mrp=spacecraft_state_output.attitude,
            shadow_factor=shadow_factor,
            battery_state_dict=battery_state_dict,
        )

        navigator_state_dict = state_dict['_navigator']
        navigator_state_dict, (sun_direction_in_body, ) = self._navigator(
            navigator_state_dict,
            position_in_inertial=spacecraft_state_output.position_in_inertial,
            mrp_attitude_in_inertial=spacecraft_state_output.attitude,
            sun_position_in_inertial=sun_position,
        )

        pointing_location_state_dict = state_dict['_pointing_location']
        pointing_location_state_dict, (
            access_state, target_position_LP_N,
            target_position_LN_N) = self._pointing_location(
                state_dict=pointing_location_state_dict,
                position_BN_N=spacecraft_state_output.position_in_inertial,
                velocity_BN_N=spacecraft_state_output.velocity_in_inertial,
                ephemeris=earth_ephemeris,
            )
        state_dict['_pointing_location'] = pointing_location_state_dict

        pointing_guide_state_dict = state_dict['_pointing_guide']
        pointing_guide_output: LocationPointingOutput
        pointing_guide_state_dict, pointing_guide_output = self._pointing_guide(
            state_dict=pointing_guide_state_dict,
            target_position_in_inertial=target_position_LN_N,
            spacecraft_position_in_inertial=spacecraft_state_output.
            position_in_inertial,
            spacecraft_attitude=spacecraft_state_output.attitude,
            spacecraft_angular_velocity_in_body=spacecraft_state_output.
            angular_velocity,
            **kwargs,
        )
        state_dict['_pointing_guide'] = pointing_guide_state_dict

        ground_mapping_state_dict = state_dict['_ground_mapping']
        self._ground_mapping()
