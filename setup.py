from setuptools import setup, find_packages

setup(
    name='SatSim',  # 包名称，需在PyPI上保持唯一
    version='0.1.0',           # 版本号，遵循语义化版本规范
    packages=find_packages(),  # 自动发现并包含所有包及子包
    python_requires='>=3.11',  # 最低Python版本要求
)
