from setuptools import setup, find_packages

setup(

    # Package definition
    name="astraviso",
    version="0.1dev.0",
    packages=find_packages(),

    # Required package data
    include_package_data=True,
    package_data={
        'astraviso': ['catalogs/*.dat'],
    },

    # Dependencies
    install_requires=["numpy", "scipy"],

    # Description
    author="Brad Sease",
    author_email="bsease@vt.edu",
    description="A python simulator for star and unresolved spacecraft imaging.",
    license="",
    keywords="star camera simulator astronomy astrometry",
    url="https://github.com/bradsease/astra-viso",

    # Install settings
    zip_safe=True,

    # Testing
    test_suite='nose.collector'
)
