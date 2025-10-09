__all__ = [
    'get_inertial_position',
    'PointingGuideStateDict',
    'PointingGuide',
    'PowerSupplyStateDict',
    'PowerSupply',
    'RemoteSensingStateDict',
    'RemoteSensing',
]
from typing import TypedDict

import torch

from satsim.architecture import Module, constants
from satsim.attitude_control.mrp_feedback import (MRPFeedback,
                                                  MRPFeedbackStateDict)
from satsim.attitude_guidance.location_pointing import (
    LocationPointing, LocationPointingOutput, LocationPointingStateDict)
from satsim.data import Battery, MRPControl, Sensor, SolarPanel
from satsim.enviroment.ground_location import AccessState
from satsim.enviroment.ground_mapping import (GroundMapping,
                                              GroundMappingStateDict)
from satsim.simulation.base import BatteryStateDict
from satsim.simulation.eclipse import compute_shadow_factor
from satsim.simulation.gravity import Ephemeris
from satsim.simulation.power import (NoBattery, SimpleBattery, SimplePowerSink,
                                     SimplePowerSinkStateDict,
                                     SimpleSolarPanel,
                                     SimpleSolarPanelStateDict)


def get_inertial_position(
    position_LP_P: torch.Tensor,
    ephemeris: Ephemeris,
) -> torch.Tensor:
    position_PN_N = ephemeris['position_CN_N']
    direction_cosine_matrix_PN = ephemeris['direction_cosine_matrix_CN']
    position_LP_N = torch.einsum(
        '...ij, ...i -> ...j',
        direction_cosine_matrix_PN,
        position_LP_P,
    )
    return position_PN_N + position_LP_N


class PointingGuideStateDict(TypedDict):
    _location_poiniting: LocationPointingStateDict
    _mrp_control: MRPFeedbackStateDict


class PointingGuide(Module[PointingGuideStateDict]):

    def __init__(
        self,
        *args,
        poiniting_direction_B_B: torch.Tensor,
        mrp_control: MRPControl,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._location_poiniting = LocationPointing(
            timer=self._timer,
            pointing_direction_B_B=poiniting_direction_B_B,
        )

        self._mrp_control = MRPFeedback(
            timer=self._timer,
            k=torch.tensor(mrp_control.k),
            ki=torch.tensor(mrp_control.ki),
            p=torch.tensor(mrp_control.p),
            integral_limit=torch.tensor(mrp_control.integral_limit),
        )

    def forward(
        self,
        state_dict: PointingGuideStateDict,
        *args,
        position_LN_N: torch.Tensor,
        position_BN_N: torch.Tensor,
        attitude_BN: torch.Tensor,
        angular_velocity_BN_B: torch.Tensor,
        reaction_wheels_speed: torch.Tensor,
        moment_of_inertia_matrix_wrt_body_point: torch.Tensor,
        moment_of_inertia_wrt_spin: torch.Tensor,
        spin_axis_in_body: torch.Tensor,
        **kwargs,
    ) -> tuple[PointingGuideStateDict, tuple[torch.Tensor]]:
        pointing_guide_state_dict = state_dict['_location_poiniting']
        pointing_guide_output: LocationPointingOutput
        pointing_guide_state_dict, pointing_guide_output = self._location_poiniting(
            state_dict=pointing_guide_state_dict,
            position_LN_N=position_LN_N,
            position_BN_N=position_BN_N,
            attitude_BN=attitude_BN,
            angular_velocity_BN_B=angular_velocity_BN_B,
        )
        state_dict['_pointing_guide'] = pointing_guide_state_dict

        mrp_control_state_dict = state_dict['_mrp_control']
        mrp_control_state_dict, (
            attitude_control_torque,
            _,
        ) = self._mrp_control(
            state_dict=mrp_control_state_dict,
            sigma_BR=pointing_guide_output.attitude_BR,
            omega_BR_B=pointing_guide_output.angular_velocity_BR_B,
            omega_RN_B=pointing_guide_output.angular_velocity_RN_B,
            domega_RN_B=torch.zeros_like(
                pointing_guide_output.angular_velocity_BR_B),
            wheel_speeds=reaction_wheels_speed,
            inertia_spacecraft_point_b_in_body=
            moment_of_inertia_matrix_wrt_body_point,
            reaction_wheels_inertia_wrt_spin=moment_of_inertia_wrt_spin,
            reaction_wheels_spin_axis=spin_axis_in_body,
        )
        state_dict['_mrp_control'] = mrp_control_state_dict

        return state_dict, (attitude_control_torque, )


class PowerSupplyStateDict(TypedDict):
    _solar_panel: SimpleSolarPanelStateDict
    _battery: BatteryStateDict


class PowerSupply(Module[PowerSupplyStateDict]):

    def __init__(
        self,
        *args,
        solar_panel: SolarPanel,
        battery: Battery,
        use_battery: bool,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._use_battery = use_battery
        if not use_battery:
            self._battery = NoBattery()
            return

        self._solar_panel = SimpleSolarPanel(
            timer=self._timer,
            panel_normal_B_B=torch.tensor(solar_panel.direction),
            panel_area=torch.tensor(solar_panel.area),
            panel_efficiency=torch.tensor(solar_panel.efficiency),
        )
        self._battery = SimpleBattery(
            timer=self._timer,
            storage_capacity=torch.tensor(battery.capacity),
            stored_charge_percentage_init=torch.tensor(battery.percentage),
        )

    def forward(
        self,
        state_dict: PowerSupplyStateDict,
        *args,
        earth_ephemeris: Ephemeris,
        sun_ephemeris: Ephemeris,
        position_BN_N: torch.Tensor,
        attitude_BN: torch.Tensor,
        **kwargs,
    ) -> tuple[PowerSupplyStateDict, tuple]:
        if not self._use_battery:
            battery_state_dict = state_dict['_battery']
            battery_state_dict, _ = self._battery(
                state_dict=battery_state_dict)
            state_dict['_battery'] = battery_state_dict
            return state_dict, tuple()

        position_SN_N = sun_ephemeris['position_CN_N']
        position_PN_N = earth_ephemeris['position_CN_N']

        shadow_factor = compute_shadow_factor(
            position_SN_N=position_SN_N,
            position_PN_N=position_PN_N,
            position_BN_N=position_BN_N,
            planet_radius=torch.tensor([constants.REQ_EARTH * 1e3]),
        )

        battery_state_dict = state_dict['_battery']
        solar_panel_state_dict = state_dict['_solar_panel']

        battery_state_dict: BatteryStateDict
        solar_panel_state_dict, (_, battery_state_dict) = self._solar_panel(
            solar_panel_state_dict,
            position_BN_N=position_BN_N,
            position_SN_N=position_SN_N,
            attitude_BN=attitude_BN,
            shadow_factor=shadow_factor,
            battery_state_dict=battery_state_dict,
        )

        battery_state_dict, _ = self._battery(state_dict=battery_state_dict)
        state_dict['_battery'] = battery_state_dict
        return state_dict, tuple()


class RemoteSensingStateDict(TypedDict):
    _ground_mapping: GroundMappingStateDict
    _power_sink: SimplePowerSinkStateDict


class RemoteSensing(Module[RemoteSensingStateDict]):

    def __init__(
        self,
        *args,
        sensor: Sensor,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._ground_mapping = GroundMapping(
            timer=self._timer,
            minimum_elevation=torch.zeros(1),
            maximum_range=torch.full([1], 1e9),
            camera_direction_B_B=torch.tensor(sensor.direction),
            half_field_of_view=torch.tensor(sensor.half_field_of_view),
        )

        self._power_sink = SimplePowerSink(
            timer=self._timer,
            power_efficiency=torch.tensor(sensor.power),
        )

    def forward(
        self,
        state_dict: RemoteSensingStateDict,
        *args,
        battery_state_dict: BatteryStateDict,
        sensor_turn_on: torch.Tensor,
        earth_ephemeris: Ephemeris,
        position_BN_N: torch.Tensor,
        velocity_BN_N: torch.Tensor,
        attitude_BN: torch.Tensor,
        position_LP_P: torch.Tensor,
        **kwargs,
    ) -> tuple[RemoteSensingStateDict, tuple[torch.Tensor, torch.Tensor]]:
        mapping_access_state: AccessState
        ground_mapping_state_dict = state_dict['_ground_mapping']
        ground_mapping_state_dict, (
            mapping_access_state,
            position_LN_N,
            position_LP_N,
        ) = self._ground_mapping(
            ephemeris=earth_ephemeris,
            position_BN_N=position_BN_N,  # [n_sc, 3]
            velocity_BN_N=velocity_BN_N,
            attitude_BN=attitude_BN,
            position_LP_P=position_LP_P,  # [n_p, 3] - p stands for mapping point
            equatorial_radius=constants.REQ_EARTH * 1e3,
            polar_radius=constants.REQ_EARTH * 1e3,
        )

        power_sink_stat_dict = state_dict['_power_sink']
        power_sink_stat_dict, (
            power_on,
            battery_state_dict,
        ) = self._power_sink(
            power_sink_stat_dict,
            turn_on=sensor_turn_on,
            battery_state_dict=battery_state_dict,
        )

        is_filming = power_on & mapping_access_state.has_access

        return state_dict, (is_filming, position_LN_N)
