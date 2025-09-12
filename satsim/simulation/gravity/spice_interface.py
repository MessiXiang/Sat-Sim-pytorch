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
    position_CN_N: Tensor  # position vector in inertial frame
    velocity_CN_N: Tensor  # velocity vector in inertial frame
    direction_cosine_matrix_CN: NotRequired[
        Tensor]  # DCM from J2000 (inertial frame) to planet-fixed
    direction_cosine_matrix_CN_dot: NotRequired[Tensor]  # DCM derivative


zero_ephemeris = lambda: Ephemeris(
    position_CN_N=torch.zeros(1, 3),
    velocity_CN_N=torch.zeros(1, 3),
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

        positions_CN_N = []
        velocities_CN_N = []
        direction_cosine_matrices_CN = []
        direction_cosine_matrices_CN_dot = []
        for gravity_body_name in names:
            gravity_body_state, _ = spiceypy.spkezr(
                gravity_body_name,
                self.j2000_current_time,
                self._reference_base,
                'NONE',
                self._zero_base,
            )
            position_CN_N = torch.tensor(gravity_body_state[:3], )
            velocity_CN_N = torch.tensor(gravity_body_state[3:], )

            gravity_body_J20002planet_fix_state = spiceypy.sxform(
                self._reference_base,
                'IAU_' + gravity_body_name,
                self.j2000_current_time,
            )
            gravity_body_J20002planet_fix_state = torch.tensor(
                gravity_body_J20002planet_fix_state, )
            direction_cosine_matrix_CN = \
                gravity_body_J20002planet_fix_state[:3, :3]
            direction_cosine_matrix_CN_dot = \
                gravity_body_J20002planet_fix_state[3:, :3]

            positions_CN_N.append(position_CN_N)
            velocities_CN_N.append(velocity_CN_N)
            direction_cosine_matrices_CN.append(direction_cosine_matrix_CN)
            direction_cosine_matrices_CN_dot.append(
                direction_cosine_matrix_CN_dot)

        return dict(), (Ephemeris(
            position_CN_N=torch.stack(positions_CN_N, dim=-2),
            velocity_CN_N=torch.stack(velocities_CN_N, dim=-2),
            direction_cosine_matrix_CN=torch.stack(
                direction_cosine_matrices_CN, dim=-3),
            direction_cosine_matrix_CN_dot=torch.stack(
                direction_cosine_matrices_CN_dot, dim=-3),
        ), )
