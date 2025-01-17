from setuptools import find_packages
from distutils.core import setup

INSTALL_REQUIRES = [
    "numpy",
    "torch>=1.21",
    "torchvision>=0.13",
    "wandb",
]

# package metadata
# List of Python modules included in the package
setup(
    name="plr_exercise",
    version="1.0.0",
    description="Package of PLR_exercise",
    author="Zilong Deng",
    author_email="dengzi@ethz.ch",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[INSTALL_REQUIRES],
    dependencies=[
        "https://download.pytorch.org/whl/torch-2.1.0+cu121-cp38-cp38-linux_x86_64.whl"
    ],
    dependency_links=[
        "https://download.pytorch.org/whl/torch-2.1.0+cu121-cp38-cp38-linux_x86_64.whl"
    ],
)
