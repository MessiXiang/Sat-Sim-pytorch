__all__ = ['RPM']
import torch

RPM = (2. * torch.pi / 60.)

# mu is calculated by G*m, where G is the gravitational constant
# and m is the mass of specified celestial body.
MU_EARTH = 398600.436
MU_SUN = 132712440023.310
