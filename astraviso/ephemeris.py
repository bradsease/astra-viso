"""
Ephemeris handling module.

References
----------
[1] Analytical Graphics, Inc., "Ephemeris File Format (*.e)".
    http://help.agi.com/stk/index.htm#stk/importfiles-02.htm
[2] Analytical Graphics, Inc., "Attitude File Format (*.a)".
    http://help.agi.com/stk/index.htm#stk/importfiles-01.htm
TODO:
    - Implement support for Lagrange and Hermite interpolation options. Code
      currently uses PCHIP and has large errors for LEOs.
    - Support for non-standard units in STK format ephemerides. Currently only
      supports m and m/s.
"""

from __future__ import division
import os
import bisect
import scipy.interpolate
import numpy as np
import datetime as dt

STK_ORBIT_FORMATS = ["EphemerisTimePos", "EphemerisTimePosVel",
                     "EphemerisTimePosVelAcc", "EphemerisLLATimePos",
                     "EphemerisLLATimePosVel", "EphemerisLLRTimePos",
                     "EphemerisLLRTimePosVel", "EphemerisMSLLLATimePos",
                     "EphemerisMSLLLATimePosVel", "EphemerisTerrainLLATimePos",
                     "EphemerisGeocentricLLATimePosVel",
                     "EphemerisGeodeticLLRTimePos",
                     "EphemerisGeodeticLLRTimePosVel", "CovarianceTimePos",
                     "CovarianceTimePosVel"]

SUPPORTED_STK_ORBIT_FORMATS = ["EphemerisTimePos", "EphemerisTimePosVel",
                               "EphemerisTimePosVelAcc"]

STK_ATT_FORMATS = ["AttitudeTimeQuaternions", "AttitudeTimeQuatScalarFirst",
                   "AttitudeTimeQuatAngVels", "AttitudeTimeAngVels",
                   "AttitudeTimeEulerAngles", "AttitudeTimeEulerAngleRates",
                   "AttitudeTimeEulerAnglesAndRates", "AttitudeTimeYPRAngles",
                   "AttitudeTimeYPRAngleRates", "AttitudeTimeYPRAnglesAndRates",
                   "AttitudeTimeDCM", "AttitudeTimeDCMAngVels",
                   "AttitudeTimeECFVector", "AttitudeTimeECIVector"]

SUPPORTED_STK_ATT_FORMATS = ["AttitudeTimeQuaternions",
                             "AttitudeTimeQuatScalarFirst",
                             "AttitudeTimeQuatAngVels", "AttitudeTimeDCM",
                             "AttitudeTimeDCMAngVels"]

class InvalidEphemerisError(Exception):
    """ Bad ephemeris file error. """
    pass
class EphemerisBoundaryError(Exception):
    """ Requested data out of bounds. """
    pass

class OrbitEphemeris:
    """
    General orbital ephemeris class.
    """

    def __init__(self, ephem_file_name=None):
        """
        OrbitEphemeris class constructor.

        Parameters
        ---------
        ephem_file_name : str
            Target ephemeris file to load.

        Returns
        -------
        orbit_ephemeris : OrbitEphemeris
            OrbitEphemeris object containing data extracted from ephem_file.
        """

        # Ephemeris variables
        self._initial_epoch = None
        self._central_body = None
        self._coord_sys = None
        self._interpolant = None

        # Load target ephemeris file
        self.load_ephemeris(ephem_file_name)

    def load_ephemeris(self, ephem_file_name):
        """
        Load an ephemeris file.

        Parameters
        ---------
        ephem_file_name : str
            Target ephemeris file to load.

        Returns
        -------
        None
        """

        # Identify ephemeris format
        ephem_format = detect_orbit_ephemeris_format(ephem_file_name)

        # Choose ephemeris processor
        processor_map = {"stk" : self.load_stk_ephemeris}
        processor_map[ephem_format](ephem_file_name)

    def load_stk_ephemeris(self, ephem_file_name):
        """
        Load an STK-format ephemeris file.

        Parameters
        ---------
        ephem_file_name : str
            Target ephemeris file to load.

        Returns
        -------
        None
        """

        # Extract file contents
        with open(ephem_file_name, 'r') as ephem_file_handle:
            ephem_file_contents = ephem_file_handle.read().split('\n')

        # Read ephemeris data
        self._initial_epoch = read_stk_orbit_epoch(ephem_file_contents)
        self._central_body = read_stk_orbit_central_body(ephem_file_contents)
        self._coord_sys = read_stk_orbit_coord_sys(ephem_file_contents)
        data_format = read_stk_orbit_data_format(ephem_file_contents)
        state_data = read_stk_orbit_state_data(ephem_file_contents, data_format)
        interp_method, interp_order = read_stk_interpolation_options(
            ephem_file_contents)
        self.validate_stk_params(data_format, state_data)

        # Build position interpolants
        x_interp = scipy.interpolate.PchipInterpolator(state_data[0],
                                                       state_data[1])
        y_interp = scipy.interpolate.PchipInterpolator(state_data[0],
                                                       state_data[2])
        z_interp = scipy.interpolate.PchipInterpolator(state_data[0],
                                                       state_data[3])

        # Combine interpolants and store
        self._interpolant = lambda time: np.array([x_interp(time),
                                                   y_interp(time),
                                                   z_interp(time)])

    def get_position(self, time):
        """
        Get ephemeris state at a specified time.

        Parameters
        ---------
        time : float, datetime
            Time of desired state. Accepts input as either seconds measured from
            the initial epoch or a specific date and time.

        Returns
        -------
        state : ndarray
            Ephemeris position at the input time.
        """

        if type(time) is dt.datetime:
            time = (time - self._initial_epoch).total_seconds()

        return self._interpolant(time)

    def validate_stk_params(self, data_format, state_data):
        """
        """

        # Check if ephemeris data format is supported
        if data_format not in SUPPORTED_STK_ORBIT_FORMATS:
            raise NotImplementedError("Unsupported ephemeris data format: {}".\
                                      format(data_format))

def detect_orbit_ephemeris_format(ephem_file_name):
    """
    Detect the format of a target ephemeris file.

    Parameters
    ----------
    ephem_file_name : str
        Target ephemeris file to identify.

    Returns
    -------
    ephem_format : str
        Short string describing the ephemeris format. Supported formats are stk
        and ccsds.

    TODO
    ----
    Implement more robust identification algorithm.
    """

    # Get file extension
    _, file_extension = os.path.splitext(ephem_file_name)

    # Map extension to file type
    extension_map = {".e" : "stk", ".oem" : "ccsds"}

    # Validate extension and return type
    if file_extension not in extension_map:
        raise NotImplementedError("Unsupported ephemeris format.")
    else:
        return extension_map[file_extension]


# STK orbit ephemeris helper functions
def read_stk_orbit_epoch(ephem_file_contents):
    """
    Read ScenarioEpoch from the contents of an STK-format ephemeris file.

    Parameters
    ----------
    ephem_file_contents : list
        Target ephemeris file contents separated into a list of lines.

    Returns
    -------
    datetime_epoch : datetime
        Extracted and parsed ScenarioEpoch.

    TODO
    ----
    Handle case where ScenarioEpoch is not required for a valid ephemeris.
    """

    # Search for initial epoch
    epoch_line = _extract_unique_line(ephem_file_contents, "ScenarioEpoch")

    # Require ScenarioEpoch to be defined
    if epoch_line is None:
        raise InvalidEphemerisError("Unable to extract ScenarioEpoch")

    # Convert ScenarioEpoch to datetime
    else:
        extracted_epoch = " ".join(epoch_line.split()[1:])
        datetime_epoch = dt.datetime.strptime(extracted_epoch,
                                              "%d %b %Y %H:%M:%S.%f")

    # Return result
    return datetime_epoch

def read_stk_orbit_central_body(ephem_file_contents):
    """
    Read CentralBody from the contents of an STK-format ephemeris file.

    Parameters
    ----------
    ephem_file_contents : list
        Target ephemeris file contents separated into a list of lines.

    Returns
    -------
    extracted_central_body : str
        Extracted central body. If the ephemeris file contains no CentralBody
        line, the default return value is "Earth".
    """

    # Search for initial epoch
    central_body_line = _extract_unique_line(ephem_file_contents, "CentralBody")

    # Default central body
    if central_body_line is None:
        extracted_central_body = "Earth"

    # Extract central body name from line
    else:
        if len(central_body_line.split()) == 2:
            extracted_central_body = central_body_line.split()[1]
        else:
            raise InvalidEphemerisError("Malformed central body line.")

    # Return result
    return extracted_central_body

def read_stk_orbit_coord_sys(ephem_file_contents):
    """
    Read CoordinateSystem from the contents of an STK-format ephemeris file.

    Parameters
    ----------
    ephem_file_contents : list
        Target ephemeris file contents separated into a list of lines.

    Returns
    -------
    extracted_coord_sys : str
        Extracted coordinate system. If the ephemeris file contains no
        CoordinateSystem line, the default return value is "Fixed".
    """

    # Search for coordinate system
    coord_sys_line = _extract_unique_line(ephem_file_contents,
                                          "CoordinateSystem")

    # Default coordinate system
    if coord_sys_line is None:
        extracted_coord_sys = "Fixed"

    # Extract coordinate system from line
    else:
        if len(coord_sys_line.split()) == 2:
            extracted_coord_sys = coord_sys_line.split()[1]
        else:
            raise InvalidEphemerisError("Malformed coordinate system line.")

    # Return result
    return extracted_coord_sys

def read_stk_orbit_data_format(ephem_file_contents):
    """
    Read ephemeris data format from the contents of an STK-format ephemeris
    file.

    Parameters
    ----------
    ephem_file_contents : list
        Target ephemeris file contents separated into a list of lines.

    Returns
    -------
    extracted_format : str
        Extracted data format.
    """

    # Extract format option]
    observed_formats = [fmt in ephem_file_contents for fmt in STK_ORBIT_FORMATS]

    # Handle errors
    if sum(observed_formats) > 1:
        raise InvalidEphemerisError("Found multiple format identifiers.")
    elif sum(observed_formats) == 0:
        raise InvalidEphemerisError("Missing or invalid data format.")

    # Return data format option
    return STK_ORBIT_FORMATS[observed_formats.index(True)]

def read_stk_interpolation_options(ephem_file_contents):
    """
    Read interpolation options from the contents of an STK-format ephemeris
    file.

    Parameters
    ----------
    ephem_file_contents : list
        Target ephemeris file contents separated into a list of lines.

    Returns
    -------
    extracted_interp_method : str
        Extracted interpolation method. Defaults to "Lagrange" if no method is
        present in the file.
    extracted_interp_order : int
        Extracted interpolation order. Defaults to 5.
    """

    # Extract method and order
    interp_method = _extract_unique_line(ephem_file_contents,
                                         "InterpolationMethod")
    interp_order = _extract_unique_line(ephem_file_contents,
                                        "InterpolationOrder")
    interp_samples_m1 = _extract_unique_line(ephem_file_contents,
                                             "InterpolationSamplesM1")

    # Handle interpolation order options
    if interp_order is not None and interp_samples_m1 is not None:
        raise InvalidEphemerisError("Found multiple interpolation order lines.")
    elif interp_order is None:
        if len(interp_samples_m1.split()) > 2:
            raise InvalidEphemerisError("Malformed InterpolationSamplesM1.")
        else:
            extracted_interp_order = interp_samples_m1.split()[1]
    elif interp_samples_m1 is None:
        if len(interp_order.split()) > 2:
            raise InvalidEphemerisError("Malformed InterpolationOrder.")
        else:
            extracted_interp_order = interp_order.split()[1]
    else:
        extracted_interp_order = 5

    # Check for invalid valid interpolation order
    try:
        extracted_interp_order = int(extracted_interp_order)
    except ValueError:
        raise InvalidEphemerisError("Malformed interpolation order: '{}'.".\
                                    format(extracted_interp_order))

    # Default method
    if interp_method is None:
        extracted_interp_method = "Lagrange"
    else:
        if len(interp_method.split()) > 2:
            raise InvalidEphemerisError("Malformed InterpolationOrder.")
        else:
            extracted_interp_method = interp_method.split()[1]

    # Return values
    return extracted_interp_method, extracted_interp_order

def read_stk_orbit_state_data(ephem_file_contents, data_format):
    """
    Read ephemeris state data from the contents of an STK-format ephemeris file.

    Parameters
    ----------
    ephem_file_contents : list
        Target ephemeris file contents separated into a list of lines.

    Returns
    -------
    ephem_data_array : ndarray
        Extracted state data. Each row in the resulting array corresponds to a
        column in the data of the ephemeris file.
    """

    # Find start / end
    start_idx = [idx for idx, line in enumerate(ephem_file_contents) if
                 data_format in line][0]
    end_idx = [idx for idx, line in enumerate(ephem_file_contents) if
               "END Ephemeris" in line][0]

    # Check for errors
    if start_idx is None or end_idx is None:
        raise InvalidEphemerisError("Unable to determine start and end of \
                                    data block.")

    # Extract data
    ephem_data = []
    for idx in range(start_idx+1, end_idx):
        split_line = ephem_file_contents[idx].split()
        if len(split_line) > 0:
            ephem_data.append([float(element) for element in split_line])

    # Convert to numpy array and return
    ephem_data_array = np.array(ephem_data)
    if len(ephem_data_array.shape) != 2:
        raise InvalidEphemerisError("Malformed data block.")
    return ephem_data_array[:, 0:7].T


class AttitudeEphemeris:
    """
    General attitude ephemeris class.
    """

    def __init__(self, ephem_file_name=None):
        """
        AttitudeEphemeris class constructor.

        Parameters
        ---------
        ephem_file_name : str
            Target ephemeris file to load.

        Returns
        -------
        attitude_ephemeris : AttitudeEphemeris
            AttitudeEphemeris object containing data extracted from ephem_file.
        """

        # Ephemeris variables
        self._initial_epoch = None
        self._coord_sys = None
        self._interpolant = None

        # Load target ephemeris file
        self.load_ephemeris(ephem_file_name)

    def load_ephemeris(self, ephem_file_name):
        """
        Load an ephemeris file.

        Parameters
        ---------
        ephem_file_name : str
            Target ephemeris file to load.

        Returns
        -------
        None
        """

        # Identify ephemeris format
        ephem_format = detect_attitude_ephemeris_format(ephem_file_name)

        # Choose ephemeris processor
        processor_map = {"stk" : self.load_stk_ephemeris}
        processor_map[ephem_format](ephem_file_name)

    def load_stk_ephemeris(self, ephem_file_name):
        """
        Load an STK-format ephemeris file.

        Parameters
        ---------
        ephem_file_name : str
            Target ephemeris file to load.

        Returns
        -------
        None
        """

        # Extract file contents
        with open(ephem_file_name, 'r') as ephem_file_handle:
            ephem_file_contents = ephem_file_handle.read().split('\n')

        # Read ephemeris data
        self._initial_epoch = read_stk_orbit_epoch(ephem_file_contents)
        self._coord_sys = read_stk_orbit_coord_sys(ephem_file_contents)
        data_format = read_stk_orbit_data_format(ephem_file_contents)
        state_data = read_stk_orbit_state_data(ephem_file_contents, data_format)
        interp_method, interp_order = read_stk_interpolation_options(
            ephem_file_contents)
        self.validate_stk_params(data_format, state_data)

        # Build position interpolants
        #x_interp = scipy.interpolate.PchipInterpolator(state_data[0],
        #                                               state_data[1])
        #y_interp = scipy.interpolate.PchipInterpolator(state_data[0],
        #                                               state_data[2])
        #z_interp = scipy.interpolate.PchipInterpolator(state_data[0],
        #                                               state_data[3])

        # Combine interpolants and store
        #self._interpolant = lambda time: np.array([x_interp(time),
        #                                           y_interp(time),
        #                                           z_interp(time)])

    def get_attitude(self, time):
        """
        Get ephemeris attitude state at a specified time.

        Parameters
        ---------
        time : float, datetime
            Time of desired state. Accepts input as either seconds measured from
            the initial epoch or a specific date and time.

        Returns
        -------
        quaternion : ndarray
            Ephemeris attitude quaternion at the input time. Output quaternion
            format designates the 4th element of the arrray as the scalar
            component.
        """

        if type(time) is dt.datetime:
            time = (time - self._initial_epoch).total_seconds()

        return self._interpolant(time)

    def validate_stk_params(self, data_format, state_data):
        """
        """

        # Check if ephemeris data format is supported
        if data_format not in SUPPORTED_STK_ATT_FORMATS:
            raise NotImplementedError("Unsupported ephemeris data format: {}".\
                                      format(data_format))

def detect_attitude_ephemeris_format(ephem_file_name):
    """
    Detect the format of a target attitude ephemeris file.

    Parameters
    ----------
    ephem_file_name : str
        Target ephemeris file to identify.

    Returns
    -------
    ephem_format : str
        Short string describing the ephemeris format. Supported formats are stk
        and ccsds.

    TODO
    ----
    Implement more robust identification algorithm.
    """

    # Get file extension
    _, file_extension = os.path.splitext(ephem_file_name)

    # Map extension to file type
    extension_map = {".a" : "stk"}

    # Validate extension and return type
    if file_extension not in extension_map:
        raise NotImplementedError("Unsupported ephemeris format.")
    else:
        return extension_map[file_extension]

def _extract_unique_line(file_contents, search_string):
    """
    """

    # Search for anchor string
    anchor_line = [line for line in file_contents if search_string in line]

    # Enforce search_string uniqueness
    if len(anchor_line) > 1:
        raise InvalidEphemerisError("Found multiple values for '{}'".format(
                                    search_string))
    elif len(anchor_line) == 0:
        anchor_line = None
    else:
        return anchor_line[0]
