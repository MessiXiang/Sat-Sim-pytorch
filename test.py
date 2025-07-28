import os.path as osp

import spiceypy
import torch

from satsim import __path__

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

gravity_body_state, _ = spiceypy.spkezr(
    'SUN',
    et,
    'J2000',
    'NONE',
    'EARTH',
)
print(gravity_body_state)
