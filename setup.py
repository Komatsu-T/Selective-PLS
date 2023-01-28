from glob import glob
from os.path import splitext
from setuptools import setup
from setuptools import find_packages

def _requires_from_file(filename):
    return open(filename).read().splitlines()

setup(
    name='selectivePLS',
    version='0.1.0',
    description='',
    author='Komatsu-T',
    url='',
    packages=find_packages(),,
    classifiers=[
            "Programming Language :: Python :: 3.6",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
)  
