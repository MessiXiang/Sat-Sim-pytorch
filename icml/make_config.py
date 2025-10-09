import json

import torch

from satsim.architecture import Timer
from satsim.data import Constellation, OrbitalElements

from .satellite import RemoteSensingConstellation

constellation = Constellation.sample(50)
timer = Timer(1.)
constellation_dict = constellation.to_dict()

orbits = OrbitalElements.sample(50)
orbits_dict = orbits.to_dicts()

latitude = torch.rand(50) * torch.pi - torch.pi / 2
longitude = torch.rand(50) * torch.pi * 2 - torch.pi
saved_config = dict(constellation=constellation_dict,
                    orbits=orbits_dict,
                    tasks=dict(
                        latitude=latitude.tolist(),
                        longitude=longitude.tolist(),
                    ))

constellation = RemoteSensingConstellation(
    timer=timer,
    constellation=constellation,
    orbits=orbits,
    use_battery=True,
)
torch.save(constellation.reset(), 'saved_state.pth')

with open('saved_config.json', 'w', encoding='utf-8') as f:
    json.dump(saved_config, f, indent=4, ensure_ascii=False)
