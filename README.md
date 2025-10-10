# Sat-Sim-pytorch
## Installation
```
pip install -i https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124
pip install todd_ai
pip install git+https://github.com/lvis-dataset/lvis-api.git@lvis_challenge_2021
pip install spiceypy
pip install pytest

pip install --no-build-isolation -e .
```

## Test
```
python -m algo.make_config
python -m algo.run_sample_enviroment
```
