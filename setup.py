from setuptools import find_packages, setup

setup(
    name="Bayesian examples",
    version="0.1",
    description="Comparison of Pymc3 and Pystan",
    author="Ryan Nazareth",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
)
