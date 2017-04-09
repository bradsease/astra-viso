"""
Astra-Viso world object module.
"""
import numpy as np
from scipy.integrate import ode
from astraviso import pointingutils

class WorldObject:
    """
    World object class.
    """

    def __init__(self):
        """
        World object initialization.
        """

        # Name
        #self.name = ""

        # Time
        self.epoch = 0                   # Currently unused
        self.epoch_format = "seconds"    # Currently unused

        # Attitude
        self.model_pointing = "on"
        #self.quaternion = np.array([0, 0, 0, 1])
        #self.angular_rate = np.array([0, 0, 0])

        # Attitude dynamics
        self.pointing_mode = "ode"       # "ode", "explicit", or "sampled"
        self.pointing_ode = None         # Storage for ode, if necessary
        self.pointing_fcn = None         # Pointing dynamics

        # Position
        self.model_position = "off"
        #self.position = np.array([0, 0, 0])
        #self.velocity = np.array([0, 0, 0])

        # Position dynamics
        self.position_mode = "ode"       # "ode", "explicit", or "sampled"
        self.position_ode = None         # Storage for ode, if necessary
        self.position_fcn = None         # Position dynamics

        # Integrator properties
        self.integrator = "dopri5"
        self.integrator_atol = 1e-9
        self.integrator_rtol = 1e-9

        # Interpolation properties
        self.interp_order = 5            # Currently unused

    def set_pointing_fcn(self, fcn, mode, initial_state=None):
        """
        Set internal pointing dynamics.
        """

        # Verify input mode
        if mode.lower() not in ["ode", "explicit"]:
            raise ValueError("Invalid pointing mode:" + mode)

        # Handle ODE option
        if mode.lower() == "ode":

            # Store ODE for later
            self.pointing_ode = fcn

            # Set up integrator and store ode
            explicit_fcn = ode(fcn)
            explicit_fcn.set_integrator(self.integrator, atol=self.integrator_atol,                \
                                                                          rtol=self.integrator_rtol)
            #explicit_fcn.set_initial_value(np.hstack((self.quaternion, self.angular_rate)), 0)
            explicit_fcn.set_initial_value(initial_state, 0)

            # Set pointing function
            pointing_fcn = explicit_fcn.integrate

        # Handle explicit option
        elif mode.lower() == "explicit":

            # Set function
            pointing_fcn = fcn

        # Verify input function
        # Integrates over a short span because setting to zero yields
        # "too small step size" warning..
        test_result = pointing_fcn(1e-6)
        if len(test_result) != 7:
            raise ValueError("Invalid pointing function output length.")
        if not isinstance(test_result, np.ndarray):
            raise TypeError("Invalid pointing function output type.")

        # Set internal pointing function and properties
        self.model_pointing = "on"
        self.pointing_mode = mode.lower()
        self.pointing_fcn = pointing_fcn

    def set_pointing_preset(self, preset, initial_state=None):
        """
        Set internal pointing dynamics to preset function.
        """

        # Rigid body kinematic option
        if preset == "kinematic":

            # Build lambda function
            function = lambda t, state: pointingutils.rigid_body_kinematic(state[0:4], state[4:])

            # Set function
            self.set_pointing_fcn(function, "ode", initial_state)

        else:
            raise NotImplementedError("Selected preset not supported.")

    def set_pointing(self, quaternion, angular_rate=np.array([0, 0, 0]), time=0):
        """
        Set initial pointing state.
            >>> MAYBE UNNECESSARY
        """

        # To be implemented...
        pass

    def get_pointing(self, time, mode="quaternion"):
        """
        Get pointing direction at a particular time.
        """

        # Ensure that time is a list
        if not isinstance(time, list) and not isinstance(time, np.ndarray):
            time = [time]
        num = len(time)

        # Collect pointing values for quaternion mode
        if mode == "quaternion":

            # Allocate
            output = np.zeros((num, 4))

            # Iterate
            for idx in range(num):
                output[idx, :] = self.pointing_fcn(time[idx])[0:4]

        # Collect pointing values for dcm mode
        elif mode == "dcm":

            # Allocate
            output = np.zeros((num, 3, 3))

            # Iterate
            for idx in range(num):
                output[idx, :, :] = pointingutils.quaternion2dcm(self.pointing_fcn(time[idx])[0:4])

        # Handle invalid mode
        else:
            raise NotImplementedError("Unsupported pointing type. Options: quaternion, dcm.")

        # Remove unnecessary dimension
        if num == 1:
            output = np.squeeze(output)

        # Return values
        return output

    def set_position_fcn(self, fcn, mode):
        """
        Set interal position dynamics.
        """

        # To be implemented...
        pass

    def get_position(self, time):
        """
        Get position at a particular time.
        """

        # To be implemented...
        pass

    def set_position_preset(self, preset):
        """
        Set internal position dynamics to preset function.
        """

        # To be implemented
        pass

    def set_integrator(self, integrator, atol=None, rtol=None):
        """
        Set internal integrator.
        """

        # Check integrator input
        accepted_values = ["vode", "isoda", "dopri5", "dop853"]
        if integrator not in accepted_values:
            raise NotImplementedError("Unsupported integrator. Options: " + str(accepted_values))

        # Set integrator
        self.integrator = integrator

        # Set absolute tolerance
        if atol is not None:
            if atol <= 0:
                raise ValueError("Integrator tolerances must be >= 0.")
            else:
                self.integrator_atol = atol

        # Set relative tolerance
        if rtol is not None:
            if rtol <= 0:
                raise ValueError("Integrator tolerances must be >= 0.")
            else:
                self.integrator_rtol = rtol

        # Update pointing function if necessary
        #if self.pointing_ode is not None:
        #    self.set_pointing_fcn(self.pointing_ode, "ode")

        # Update position function if necessary
        #if self.position_ode is not None:
        #    self.set_position_fcn(self.position_ode, "ode")
