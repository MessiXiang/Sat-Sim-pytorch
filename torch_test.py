from satsim.utils.orbital_motion import OrbitalElements, elem2rv
from satsim.architecture import constants
import torch

torch.set_default_dtype(torch.float64)
oe = OrbitalElements(
    **{
        "eccentricity": 0.0003419,
        "semi_major_axis": 6879000.0,
        "inclination": 97.4975 * (torch.pi / 180.),
        "longitude_of_the_ascending_node": 117.0006 * (torch.pi / 180.),
        "arguments_of_periapsis": 75.9277 * (torch.pi / 180.),
        "true_anomaly": 289.1165731302157 * (torch.pi / 180.)
    })

r, v = elem2rv(constants.MU_EARTH, oe)
print(r.tolist())
print(v.tolist())
