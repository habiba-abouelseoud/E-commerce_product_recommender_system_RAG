from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='e-commerce_reccomender_sys',
    version='0.1',
    author='Habibatallah Abouelseoud',
    packages= find_packages(),
    install_requires = requirements,

)