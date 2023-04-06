from setuptools import setup, find_packages

# read in the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='pyGridSampler',
    description='Adaptive grid-based sampling with iterative batch sizes.',
    version='0.0.1',
    packages=find_packages(),
    install_requires=requirements,
)
