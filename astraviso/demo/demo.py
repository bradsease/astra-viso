"""
Astra Viso demos.
"""

import astraviso as av
import numpy as np

def default():
    """
    Default Astra Viso demo. Creates a single image.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    # Create StarCam and define motion
    cam = av.StarCam()
    cam.set_pointing_preset("kinematic", initial_quaternion=[0,0,0,1,],                            \
                                                                 initial_angular_rate=[0,0.025,0.1])

    # Setup star catalog
    cam.star_catalog.load_preset("random", 100000)

    # Create WorldObject and define motion
    cam.add_worldobject()
    cam.external_objects[0].set_position_preset("kinematic",                                       \
                 initial_position=np.array([25, -50, 1000]), initial_velocity=np.array([50, 50, 0]))
    cam.external_objects[0].set_vismag_preset("constant", vismag=2)

    # Create image and display
    av.imageutils.imshow(cam.snap(0, 1), [])

def sequence(count=2):
    """
    Astra Viso sequence demo.

    Parameters
    ----------
    count : int, optional
        Number of sequential images to generate. Default is 2.

    Returns
    -------
    None
    """

    # Create StarCam and define motion
    cam = av.StarCam()
    cam.set_pointing_preset("kinematic", initial_quaternion=[0,0,0,1,],                            \
                                                                  initial_angular_rate=[0.01, 0, 0])

    # Setup star catalog
    cam.star_catalog.load_preset("random", 100000)

    # Create WorldObject and define motion
    cam.add_worldobject()
    cam.external_objects[0].set_position_preset("kinematic",                                       \
                         initial_position=np.array([0, 0, 1]), initial_velocity=np.array([0, 0, 0]))
    cam.external_objects[0].set_vismag_preset("constant", vismag=-1)

    # Create image and display
    for idx in range(count):
        av.imageutils.imshow(cam.snap(idx, 1), [])
