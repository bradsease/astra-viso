"""
Run unit tests
"""
import numpy as np
import astraviso as av

cam = av.StarCam()
cam.add_worldobject()
cam.external_objects[0].set_position_preset("kinematic", initial_position=np.array([0, 0, 1000]), initial_velocity=np.array([100, 50, 0]))
cam.add_worldobject()
cam.external_objects[1].set_position_preset("kinematic", initial_position=np.array([-50, 0, 1000]), initial_velocity=np.array([-100, -50, 0]))
cam.set_pointing_preset("kinematic", [0,0,0,1,0,0.025,0.1])
cam.star_catalog.load_preset("random", 10000)
cam.set_quantum_efficiency_preset("gaussian", quantum_efficiency=0.22, sigma=0.01, seed=1)
av.imageutils.imshow(cam.snap(0, 1), [1420, 1600])
#av.imageutils.imshow(cam.snap(1, 1))