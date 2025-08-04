__all__ = ['RPM']
import torch

RPM = (2. * torch.pi / 60.)

UTC_TIME_START = '2019-01-01T12:00:00'

# Celestial Body ()
# Gm. All units are km^3/s^2.
MU_SUN = 132712440023.310
MU_EARTH = 398600.436
REQ_SUN = 695000.
REQ_EARTH = 6378.1366
