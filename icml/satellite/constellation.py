__all__ = [
    'RemoteSensingConstellationStateDict',
    'RemoteSensingConstellation',
]
from typing import TypedDict, cast

import torch
import torch.nn.functional as F

from satsim.architecture import Module, constants
from satsim.data import (Constellation, OrbitalElements, ReactionWheelGroups,
                         elem2rv)
from satsim.simulation.gravity import (Ephemeris, GravityField,
                                       PointMassGravityBody, SpiceInterface)
from satsim.simulation.reaction_wheels import (HoneywellHR12Small,
                                               ReactionWheels, concat)
from satsim.simulation.spacecraft import (IntegrateMethod, Spacecraft,
                                          SpacecraftStateDict,
                                          SpacecraftStateOutput)
from satsim.utils import move_to

from .components import (PointingGuide, PointingGuideStateDict, PowerSupply,
                         PowerSupplyStateDict, RemoteSensing,
                         RemoteSensingStateDict)


class RemoteSensingConstellationStateDict(TypedDict):
    _spacecraft: SpacecraftStateDict
    _power_supply: PowerSupplyStateDict
    _pointing_guide: PointingGuideStateDict
    _sun_guide: PointingGuideStateDict
    _remote_sensing: RemoteSensingStateDict


class RemoteSensingConstellation(Module[RemoteSensingConstellationStateDict]):

    def __init__(
        self,
        *args,
        constellation: Constellation,
        orbits: OrbitalElements | None = None,
        use_battery: bool = True,
        integrate_method: IntegrateMethod = 'RK',
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._n = len(constellation.mass)

        self._setup_spacecraft(constellation, integrate_method, orbits=orbits)
        self._setup_power_supply(constellation, use_battery)
        self._setup_pointing_guide(constellation)
        self._setup_remote_sensing(constellation)

    @property
    def num_satellite(self) -> int:
        return self._n

    @property
    def spice_interface(self) -> SpiceInterface:
        return cast(SpiceInterface,
                    self._spacecraft.gravity_field.spice_interface)

    @property
    def reaction_wheels(self) -> ReactionWheels:
        return cast(ReactionWheels, self._spacecraft.reaction_wheels)

    def _setup_gravity_field(self) -> GravityField:
        sun = PointMassGravityBody.create_sun(timer=self._timer)
        earth = PointMassGravityBody.create_earth(
            timer=self._timer,
            is_central=True,
        )

        spice_interface = SpiceInterface(
            timer=self._timer,
            utc_time_init=constants.UTC_TIME_START,
        )

        return GravityField(
            timer=self._timer,
            spice_interface=spice_interface,
            gravity_bodies=[sun, earth],
        )

    def _setup_spacecraft(
        self,
        constellation: Constellation,
        integrate_method: IntegrateMethod,
        orbits: OrbitalElements | None = None,
    ) -> None:
        if orbits is None:
            orbits = OrbitalElements.sample(self._n)

        gravity_field = self._setup_gravity_field()
        reaction_wheels = self._setup_reaction_wheels(
            constellation.reaction_wheels)

        r, v = elem2rv(constants.MU_EARTH * 1e9, elements=orbits)
        inertias = torch.tensor(constellation.inertia).view(-1, 3, 3)
        self._spacecraft = Spacecraft(
            timer=self._timer,
            mass=torch.tensor(constellation.mass),
            moment_of_inertia_matrix_wrt_body_point=inertias,
            position_BP_N=r,
            velocity_BP_N=v,
            angular_velocity_BN_B=torch.zeros(self._n, 3),
            gravity_field=gravity_field,
            reaction_wheels=reaction_wheels,
            integrate_method=integrate_method,
        )

    def _setup_reaction_wheels(
        self,
        reaction_wheels_groups: ReactionWheelGroups,
    ) -> ReactionWheels:
        reaction_wheels_0 = []
        reaction_wheels_1 = []
        reaction_wheels_2 = []

        for rw0, rw1, rw2 in reaction_wheels_groups:
            reaction_wheels_0.append(
                HoneywellHR12Small.build(
                    mech_to_elec_efficiency=rw0.efficiency,
                    base_power=rw0.power,
                    angular_velocity_init=rw0.rw_speed_init,
                ))
            reaction_wheels_1.append(
                HoneywellHR12Small.build(
                    mech_to_elec_efficiency=rw1.efficiency,
                    base_power=rw1.power,
                    angular_velocity_init=rw1.rw_speed_init,
                ))
            reaction_wheels_2.append(
                HoneywellHR12Small.build(
                    mech_to_elec_efficiency=rw2.efficiency,
                    base_power=rw2.power,
                    angular_velocity_init=rw2.rw_speed_init,
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

    def _setup_power_supply(
        self,
        constellation: Constellation,
        use_battery: bool,
    ) -> None:
        self._use_battery = use_battery
        self._power_supply = PowerSupply(
            timer=self._timer,
            solar_panel=constellation.solar_panel,
            battery=constellation.battery,
            use_battery=use_battery,
        )

    def _setup_pointing_guide(self, constellation: Constellation) -> None:
        if self._use_battery:
            self._sun_guide = PointingGuide(
                timer=self._timer,
                poiniting_direction_B_B=constellation.solar_panel.direction,
                mrp_control=constellation.mrp_control,
            )

        self._pointing_guide = PointingGuide(
            timer=self._timer,
            poiniting_direction_B_B=constellation.sensor.direction,
            mrp_control=constellation.mrp_control,
        )

    def _setup_remote_sensing(self, constellation: Constellation) -> None:
        self._remote_sensing = RemoteSensing(
            timer=self._timer,
            sensor=constellation.sensor,
        )

    def _simple_motor_torque_assign(
        self,
        command_torque: torch.Tensor,
    ) -> torch.Tensor:
        return -command_torque

    def _update_power_system(
        self,
        state_dict: RemoteSensingConstellationStateDict,
        motor_torque: torch.Tensor,
        earth_ephemeris: Ephemeris,
        sun_ephemeris: Ephemeris,
        position_BN_N: torch.Tensor,
    ) -> RemoteSensingConstellationStateDict:
        spacecraft_state_dict = state_dict['_spacecraft']
        attitude_BN = spacecraft_state_dict['_hub']['dynamic_params'][
            'attitude_BN']

        reaction_wheels_state_dict = state_dict['_spacecraft'][
            '_reaction_wheels']
        reaction_wheels_state_dict, (
            battery_state_dict, ) = self.reaction_wheels(
                state_dict=reaction_wheels_state_dict,
                battery_state_dict=battery_state_dict,
                motor_torque=motor_torque,
            )
        state_dict['_spacecraft'][
            '_reaction_wheels'] = reaction_wheels_state_dict

        power_supply_state_dict = state_dict['_power_supply']
        power_supply_state_dict, _ = self._power_supply(
            power_supply_state_dict,
            earth_ephemeris=earth_ephemeris,
            sun_ephemeris=sun_ephemeris,
            position_BN_N=position_BN_N,
            attitude_BN=attitude_BN,
        )
        state_dict['_power_supply'] = power_supply_state_dict

        return state_dict

    def _calculate_motor_torque(
        self,
        state_dict: RemoteSensingConstellationStateDict,
        position_LN_N: torch.Tensor,
        position_BN_N: torch.Tensor,
        attitude_BN: torch.Tensor,
        angular_velocity_BN_B: torch.Tensor,
        charge: torch.Tensor | None = None,
    ) -> tuple[RemoteSensingConstellationStateDict, torch.Tensor]:

        pointing_guide_state_dict = state_dict['_pointing_guide']
        sun_guide_state_dict, (command_torque, ) = self._sun_guide(
            pointing_guide_state_dict,
            position_LN_N,
            position_BN_N,
            attitude_BN,
            angular_velocity_BN_B,
            self._spacecraft._hub.moment_of_inertia_matrix_wrt_body_point,
            self.reaction_wheels.moment_of_inertia_wrt_spin,
            self.reaction_wheels.spin_axis_in_body,
        )

        if self._use_battery:
            if charge is None:
                raise ValueError(
                    "when battery is used, you must determine whether to charge"
                )
            sun_guide_state_dict = state_dict['_sun_guide']
            sun_guide_state_dict, (
                sun_guide_command_torque,
            ) = self._sun_guide(
                sun_guide_state_dict,
                position_LN_N,
                position_BN_N,
                attitude_BN,
                angular_velocity_BN_B,
                self._spacecraft._hub.moment_of_inertia_matrix_wrt_body_point,
                self.reaction_wheels.moment_of_inertia_wrt_spin,
                self.reaction_wheels.spin_axis_in_body,
            )

            command_torque = torch.where(
                charge.unsqueeze(-1),
                sun_guide_command_torque,
                command_torque,
            )

        return state_dict, self._simple_motor_torque_assign(command_torque)

    def _verify_sensing_acquisition(
        self,
        state_dict: RemoteSensingConstellationStateDict,
        sensor_turn_on: torch.Tensor,
        earth_ephemeris: Ephemeris,
        position_BN_N: torch.Tensor,
        velocity_BN_N: torch.Tensor,
        attitude_BN: torch.Tensor,
        position_LP_P: torch.Tensor,
    ) -> tuple[RemoteSensingStateDict, torch.Tensor, torch.Tensor]:
        remote_sensing_state_dict = state_dict['_remote_sensing']
        battery_state_dict = state_dict['_power_supply']['_battery']
        remote_sensing_state_dict, (
            is_filming,
            position_LN_N,
        ) = self._remote_sensing(
            remote_sensing_state_dict,
            battery_state_dict=battery_state_dict,
            sensor_turn_on=sensor_turn_on,
            earth_ephemeris=earth_ephemeris,
            position_BN_N=position_BN_N,
            velocity_BN_N=velocity_BN_N,
            attitude_BN=attitude_BN,
            position_LP_P=position_LP_P,
        )
        return state_dict, is_filming, position_LN_N

    def forward(
        self,
        state_dict: RemoteSensingConstellationStateDict,
        *args,
        charging: torch.Tensor | None = None,
        sensor_turn_on: torch.Tensor,
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
        battery_state_dict = state_dict['_power_supply']['_battery']

        sun_ephemeris: Ephemeris
        earth_ephemeris: Ephemeris
        _, (sun_ephemeris, ) = self.spice_interface(names=['SUN'])
        _, (earth_ephemeris, ) = self.spice_interface(names=['EARTH'])
        sun_ephemeris = move_to(
            sun_ephemeris,
            attitude_BN,
        )
        earth_ephemeris = move_to(
            earth_ephemeris,
            attitude_BN,
        )
        state_dict, is_filming, position_LN_N = self._verify_sensing_acquisition(
            state_dict,
            sensor_turn_on,
            earth_ephemeris,
            spacecraft_state_output.position_BN_N,
            spacecraft_state_output.velocity_BN_N,
            attitude_BN,
            self._position_LP_P,
        )

        state_dict, motor_torque = self._calculate_motor_torque(
            state_dict,
            position_LN_N,
            spacecraft_state_output.position_BN_N,
            attitude_BN,
            angular_velocity_BN_B,
            charging,
        )

        state_dict = self._update_power_system(
            state_dict,
            motor_torque,
            earth_ephemeris,
            sun_ephemeris,
            spacecraft_state_output,
        )

        return state_dict, (
            is_filming,
            spacecraft_state_output,
            battery_state_dict['stored_charge_percentage'],
        )

    def setup_target(
        self,
        state_dict: RemoteSensingConstellationStateDict,
        position_LP_P: torch.Tensor,
    ) -> RemoteSensingConstellationStateDict:
        self._position_LP_P = position_LP_P

        return state_dict
