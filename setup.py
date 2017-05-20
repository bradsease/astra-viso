"""
Astra Viso is a full-featured toolset for simulating stars and unresolved
space objects. With this software it is simple to set up a simulated sensor
and begin generating imagery immediately. The tools allow the user to switch
between a number of commonly-used internal modeling presets easily while
also having the flexibility to override internal modeling with externally-
defined functions.
"""
from setuptools import setup, find_packages

DOCLINES = (__doc__ or '').split("\n")

setup(

    # Package definition
    name="astraviso",
    version="0.1.1",
    packages=find_packages(),

    # Required package data
    include_package_data=True,
    package_data={
        'astraviso': ['catalogs/*.dat'],
    },

    # Dependencies
    install_requires=["numpy", "scipy", "matplotlib", "numba"],

    # Description
    author="Brad Sease",
    author_email="bsease@vt.edu",
    description="A python simulator for star and unresolved spacecraft imaging.",
    long_description="\n".join(DOCLINES[1:]),
    license="MIT",
    keywords="star camera simulator astronomy astrometry",
    url="https://github.com/bradsease/astra-viso",

    # Install settings
    zip_safe=True,

    # Testing
    test_suite='nose.collector'
)
