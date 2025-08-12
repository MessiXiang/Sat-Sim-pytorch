# Sat-Sim-pytorch

If no frame specified, assume it to be in inertial frame(J2000) other than angular velocity. 
if no frame specified, angular velocity is assumed to be observed from inertial frame in its own frame. \
e.g.
(In Spacecraft Module) `angular_velocity` means angular velocity observed from inertial frame in body frame.
## Installation
```
pip install torch torchaudio torchvision
pip install todd_ai
pip install git+https://github.com/lvis-dataset/lvis-api.git@lvis_challenge_2021
pip install pytest

pip install --no-build-isolation -e .
```

## Test
```
pytest -vv .
```