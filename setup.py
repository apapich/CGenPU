#!/usr/bin/env python
import setuptools

def readme():
    with open("README.rst", encoding="utf-8") as f:
        return f.read()

__version__: str = "0.1"

setuptools.setup(
    name="CGenPU",
    description="Conditional Generative PU framework",
    long_description=readme(),
    version=__version__,
    license="BSD-3-Clause",

    author="Aleš Papič",
    author_email="ales.papic@fri.uni-lj.si",
    url="https://github.com/apapich/CGenPU",

    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

    packages=setuptools.find_packages(include=["pu"]),
    python_requires='>=3.8',
    install_requires=[
        "matplotlib==3.3.1",
        "numpy==1.19.5",
        "tensorflow-gpu==2.4.0",
        "tqdm==4.58.0",
        "wandb==0.10.21"
    ]
)