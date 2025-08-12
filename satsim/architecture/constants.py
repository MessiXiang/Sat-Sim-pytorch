__all__ = ['RPM']
import torch

RPM = (2. * torch.pi / 60.)

# Celestial Body
# Gm. All units are km^3/s^2.

MU_SUN = 132712440023.310
MU_EARTH = 398600.436
REQ_SUN = 695000.
REQ_EARTH = 6378.1366

# Physical constants
SOLAR_FLUX_AT_EARTH = 1367.0  # (W/m²)
ASTRONOMICAL_UNIT = 149597870.7 * 1000.0  # m

# System constants
UTC_TIME_START = '2019-01-01T12:00:00'

# Algorithm parameters

# LocationPointing
PARALLEL_TOLERANCE = 0.1
