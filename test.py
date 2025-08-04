import os.path as osp

import numpy as np
import spiceypy
import torch

from satsim import __path__
from satsim.architecture import constants

file_names = [
    "de430.bsp",
    "naif0012.tls",
    "de-403-masses.tpc",
    "pck00010.tpc",
]
path = list(__path__)[0]
kernel_files = [
    osp.join(path, 'spice_kernel', file_name) for file_name in file_names
]
spiceypy.furnsh(kernel_files)

utc_time = '2010-07-25T12:00:00'
et = spiceypy.utc2et(utc_time)

states = spiceypy.conics(
    np.array([
        6879000.0,
        0.0003419,
        97.4975 * (torch.pi / 180.),
        117.0006 * (torch.pi / 180.),
        75.9277 * (torch.pi / 180.),
        289.1165731302157 * (torch.pi / 180.),
        0.,
        constants.MU_EARTH,
    ]), et)

print(states[:3])
print(states[3:])
