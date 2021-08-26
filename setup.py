from setuptools import find_packages, setup

install_requires = [
    'torch==1.8.0',
    'torchaudio==0.8.0',
    'numpy',
    'torch-audiomentations',
]

setup(
    name='music-recognition',
    version='0.1.0',
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.6,<3.9",
)
