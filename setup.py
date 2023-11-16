#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='mpf',
    version='0.0.1',
    url='https://github.com/mpiraux/mpf.git',
    author='Maxime Piraux',
    author_email='maxime.piraux@uclouvain.be',
    description='Minimal Performance Framework',
    packages=find_packages(),    
    install_requires=[
        'ipyparallel >= 8.6', 
        'pandas >= 1.5',
        'PyYAML >= 6.0',
        'scipy >= 1.11',
        'tqdm'
        ],
)