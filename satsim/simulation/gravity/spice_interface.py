__all__ = ['Ephemeris', 'SpiceInterface']

import os.path as osp
from typing import Iterable, NotRequired, TypedDict

import spiceypy
import torch
from torch import Tensor

from satsim.architecture import Module, VoidStateDict

from . import __path__


def string_normalizer(s: str):
    return s.strip().upper().replace(' ', '_')


class Ephemeris(TypedDict):
    """
    All Tensor should be of shape [batch_size, num_body, *value_shape]. First two dimension can be ommited or set to 1
    
    """
    position_in_inertial: Tensor  # position vector
    velocity_in_inertial: Tensor  # velocity vector
    J2000_2_planet_fixed: NotRequired[Tensor]  # DCM from J2000 to planet-fixed
    J2000_2_planet_fixed_dot: NotRequired[Tensor]  # DCM derivative


zero_ephemeris = lambda: Ephemeris(
    position_in_inertial=torch.zeros(1, 3),
    velocity_in_inertial=torch.zeros(1, 3),
)


class SpiceInterface(Module[VoidStateDict]):

    def __init__(
        self,
        *args,
        utc_time_init: str,
        kernel_files: Iterable[str] | None = None,
        zero_base: str = 'Earth',
        reference_base: str = 'J2000',
        **kwagrs,
    ) -> None:
        super().__init__(*args, **kwagrs)
        if kernel_files is None:
            file_names = [
                "de430.bsp",
                "naif0012.tls",
                "de-403-masses.tpc",
                "pck00010.tpc",
            ]
            path = list(__path__)[0]
            kernel_files = [
                osp.join(path, 'spice_kernel', file_name)
                for file_name in file_names
            ]
        spiceypy.furnsh(kernel_files)

        self._zero_base = string_normalizer(zero_base)

        self._utc_time_init = utc_time_init
        self._ephemeris_time_init = spiceypy.utc2et(utc_time_init)
        self._reference_base = reference_base

    @property
    def j2000_current_time(self):
        return self._ephemeris_time_init + self._timer.time

    @property
    def zero_base(self):
        return self._zero_base

    def forward(
        self,
        names: str | list[str],
        state_dict: VoidStateDict | None = None,
        *args,
        **kwargs,
    ) -> tuple[VoidStateDict, tuple[Ephemeris]]:

        if isinstance(names, str):
            names = [names]

        gravity_body_positions_in_inertial = []
        gravity_body_velocities_in_inertial = []
        gravity_body_J20002planet_fix_matrices = []
        gravity_body_J20002planet_fix_matrices_dot = []
        for gravity_body_name in names:
            gravity_body_state, _ = spiceypy.spkezr(
                gravity_body_name,
                self.j2000_current_time,
                self._reference_base,
                'NONE',
                self._zero_base,
            )
            gravity_body_position_in_inertial = torch.tensor(
                gravity_body_state[:3], )
            gravity_body_velocity_in_inertial = torch.tensor(
                gravity_body_state[3:], )

            gravity_body_J20002planet_fix_state = spiceypy.sxform(
                self._reference_base,
                'IAU_' + gravity_body_name,
                self.j2000_current_time,
            )
            gravity_body_J20002planet_fix_state = torch.tensor(
                gravity_body_J20002planet_fix_state, )
            gravity_body_J20002planet_fix_matrix = \
                gravity_body_J20002planet_fix_state[:3, :3]
            gravity_body_J20002planet_fix_matrix_dot = \
                gravity_body_J20002planet_fix_state[3:, :3]

            gravity_body_positions_in_inertial.append(
                gravity_body_position_in_inertial)
            gravity_body_velocities_in_inertial.append(
                gravity_body_velocity_in_inertial)
            gravity_body_J20002planet_fix_matrices.append(
                gravity_body_J20002planet_fix_matrix)
            gravity_body_J20002planet_fix_matrices_dot.append(
                gravity_body_J20002planet_fix_matrix_dot)

        return dict(), (Ephemeris(
            position_in_inertial=torch.stack(
                gravity_body_positions_in_inertial, dim=-2),
            velocity_in_inertial=torch.stack(
                gravity_body_velocities_in_inertial, dim=-2),
            J2000_2_planet_fixed=torch.stack(
                gravity_body_J20002planet_fix_matrices, dim=-3),
            J2000_2_planet_fixed_dot=torch.stack(
                gravity_body_J20002planet_fix_matrices_dot, dim=-3),
        ), )
