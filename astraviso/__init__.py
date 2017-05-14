"""
Astra Viso.
"""

from . import demo
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

def run_demo(name="default"):
    """
    Run a built-in Astra Viso demo. Current options are:

    "default" -- Default Astra Viso demo. Creates a single image.

    Parameters
    ----------
    name : str, optional
        Name of the demo to run.

    Returns
    -------
    None
    """

    # Import demo module
    from . import demo

    # Run chosen demo
    if name.lower() == "default":
        demo.default()

    # Handle invalid demo name
    else:
        raise NotImplementedError("Chosen demo is not valid.")
