"""
Astra Viso.
"""

from .starcam import StarCam
from .starmap import StarMap
from .worldobject import WorldObject

def verify(verbosity=1):
    """
    Verify Astra Viso installation by running all internal tests.

    Parameters
    ----------
    verbosity : int
        Desired output level for testing. Default is 1.

    Returns
    -------
    success : bool
        Result of testing. Returns false for any failures.
    """

    # Import tests
    import unittest as ut
    from . import test

    # Set up test suite
    suite = ut.defaultTestLoader.loadTestsFromModule(test)
    runner = ut.TextTestRunner(verbosity=verbosity)

    # Run tests
    return runner.run(suite).wasSuccessful()
