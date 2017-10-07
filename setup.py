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
    version="0.1.3",
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

    # Classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',

        # Pick your license as you wish (should match "license" above)
         'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    # Install settings
    zip_safe=True,

    # Testing
    test_suite='nose.collector'
)
