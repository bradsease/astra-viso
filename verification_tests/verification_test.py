"""
StarCam unit tests.
"""
import unittest
import astraviso as av
import numpy as np

class verificationtests(unittest.TestCase):
    """
    Astra Viso verification tests.

    Requirements Verified
    ---------------------
    1) Objects along the boresight of the sensor must always appear in the image
       for when using the pinhole projection model, regardless of the pointing
       direction.
    2) An object aligned with the boresight of the sensor and moving along the
       x or y axis in intertial space must move along the same axis in the image
       when using the pinhole model.
    """

    def setUp(self):
        self.starcam = av.StarCam()

    def tearDown(self):
        del self.starcam

class test_pointing_consistency(verificationtests):
    """
    Verify StarCam and WorldObject pointing consistency.
    """

    def test_random(self):
        """
        Test multiple randomly-generated orientations.
        """

        # Initial setup
        num_tests = 25
        cam = av.StarCam()
        cam.star_catalog.load_preset("random", 0)
        cam.set_noise_preset("off")

        # Speed-up hack for when integration accuracy is not important
        cam._StarCam__settings["integration_steps"] = 1

        # Iterate through test cases
        for idx in range(num_tests):

            # Generate random quaternion
            rand_quat = np.random.rand(4)
            rand_quat = rand_quat / np.linalg.norm(rand_quat)

            # Set up camera
            cam.set_pointing_preset("static", initial_quaternion=rand_quat)

            # Set up object
            obj = av.WorldObject()
            position = 10*cam.get_pointing(0, mode="dcm")[:,2]
            obj.set_position_preset("kinematic", initial_position=position,                        \
                                                               initial_velocity=np.array([0, 0, 0]))

            # Add object and generate image
            cam.add_worldobject(obj)
            image = cam.integrate(0, 1)

            # Verify requirement #1
            self.assertGreater(np.sum(image), 0, "Image must contain test object.")
