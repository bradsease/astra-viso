"""
Astra-Viso star camera module.
"""
from __future__ import division
import numpy as np
from scipy import signal
from astraviso import worldobject
from astraviso import starmap
from astraviso import imageutils
from astraviso import projectionutils
from astraviso import positionutils

class StarCam(worldobject.WorldObject):
    """
    Star camera class.
    """

    def __init__(self):
        """
        StarCam initialization.

        Parameters
        ----------
        None

        Returns
        -------
        starcam : StarCam
            Default StarCam object.
        """

        # Internal settings
        self.__settings = {}
        self.__settings["resolution"] = 1024
        self.__settings["max_angle_step"] = 1e-4
        self.__settings["integration_steps"] = 1000

        # Set psf size
        self.setpsf(7, 1) # To be removed...

        # Internal function variables
        self.sensitivity_fcn = None
        self.projection_fcn = None
        self.quantum_efficiency_fcn = None
        self.noise_fcn = None
        self.saturation_fcn = None

        # Set star catalog defaults
        self.star_catalog = starmap.StarMap()
        self.star_catalog.load_preset("random", 10000)

        # Set sensor pointing default
        worldobject.WorldObject.__init__(self)
        self.set_pointing_preset("kinematic", initial_quaternion=np.array([0, 0, 0, 1]),           \
                                                           initial_angular_rate=np.array([0, 0, 0]))

        # Set position model default
        self.set_position_preset("kinematic", initial_position=np.array([0, 0, 0]),                \
                                                               initial_velocity=np.array([0, 0, 0]))

        # Projection model defaults
        self.set_projection_preset("pinhole", focal_len=93, pixel_size=0.016, resolution=1024)

        # Set CCD defaults
        self.set_saturation_preset("no_bleed", bit_depth=16)
        self.set_quantum_efficiency_preset("constant", quantum_efficiency=0.22)
        self.set_noise_preset("poisson", dark_current=1200, read_noise=200)
        self.set_sensitivity_preset("default", aperture=1087, mv0_flux=19000)

        # External objects
        self.external_objects = []

    def setpsf(self, size, sigma):
        """
        Set PSF to Gaussian kernel.

        In the future should have a separate function to handle explicit
        PSF definitions.
        """

        # Enforce odd dimensions
        if size % 2 == 0:
            size = size + 1

        # Allocate variables
        halfwidth = (size-1)/2
        kernel = np.zeros((size, size))

        # Create kernel
        for row in range(size):
            for col in range(size):
                kernel[row, col] = np.exp(-0.5 * ((row-halfwidth)**2 +                   \
                                                           (col-halfwidth)**2) / sigma**2)

        # Normalize and return
        self.psf = kernel / np.sum(kernel)

    def get_boresight(self, time):
        """
        Extract boresight pointing vector at a given time.

        Parameters
        ----------
        time : float
            Time, in seconds, to compute boresight vector.

        Returns
        -------
        boresight : ndarray
            Unit vector describing the orientation of the sensor boresight in
            the default intertial frame.

        Examples
        --------
        >>> cam = StarCam()
        >>> cam.get_boresight(0)
        array([ 0.,  0.,  1.])
        """
        return self.get_pointing(time, mode="dcm")[:,2]

    def add_worldobject(self, obj=None):
        """
        Add new or existing WorldObject to the external object catalog.

        Parameters
        ----------
        obj : WorldObject, optional
            An existing WorldObject instance. If obj is not defined, a new
            WorldObject instance will be created.

        Returns
        -------
        None

        Notes
        -----
        The user can manage elements in the external object catalog directly
        through the external_objects property of the StarCam class.

        Examples
        --------
        >>> cam = StarCam()
        >>> obj1 = WorldObject()
        >>> cam.add_worldobject(obj1)
        >>> cam.external_objects
        [<astraviso.worldobject.WorldObject object at 0x000001A4C8AA6438>]
        >>> cam.external_objects[0].set_pointing_preset("kinematic",           \
                                                      np.array([0,0,0,1,0,0,0]))
        """

        # Append input WorldObject
        if isinstance(obj, worldobject.WorldObject):
            self.external_objects.append(obj)

        # Append new WorldObject
        elif obj is None:
            self.external_objects.append(worldobject.WorldObject())

        # Handle invalid input
        else:
            raise ValueError("Input must either be an existing WorldObject or None.")

    def delete_worldobject(self, index):
        """
        Clear a WorldObject from the external object catalog.

        Parameters
        ----------
        index : int
            Index of the WorldObject catalog element to remove.

        Returns
        -------
        None

        Notes
        -----
        The user can manage elements in the external object catalog directly
        through the external_objects property of the StarCam class.

        Examples
        --------
        >>> cam = StarCam()
        >>> obj1 = WorldObject()
        >>> cam.add_worldobject(obj1)
        >>> cam.external_objects
        [<astraviso.worldobject.WorldObject object at 0x000001A4C8AA6438>]
        >>> cam.delete_worldobject(0)
        >>> cam.external_objects
        []
        """

        # Delete object
        del self.external_objects[index]

    def display_scene(self):
        """
        Display a 3D plot of the sensor and all relevant worldobjects in the
        default inertial frame.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        raise NotImplementedError('Not yet implemented.')

    def integrate(self, time, delta_t):
        """
        Compute CCD pixel values after set exposure time.

        Parameters
        ----------
        time : float
            Time to begin exposure. Measured in seconds from epoch.
        delta_t : float
            Desired exposure time. Measured in seconds.

        Returns
        -------
        img : ndarray
            Resulting CCD array values. Each element contains a photon count.

        Examples
        --------
        >>> cam = StarCam()
        >>> cam.integrate(1)
        array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
               [ 0.,  0.,  0., ...,  0.,  0.,  0.],
               [ 0.,  0.,  0., ...,  0.,  0.,  0.],
               ...,
               [ 0.,  0.,  0., ...,  0.,  0.,  0.],
               [ 0.,  0.,  0., ...,  0.,  0.,  0.],
               [ 0.,  0.,  0., ...,  0.,  0.,  0.]])
        """

        # Determine step size
        # Temporary solution...
        steps = self.__settings["integration_steps"]
        step_size = delta_t / steps
        angle = 0

        # Extract subset of stars from catalog
        # Also a temporary solution...
        field_of_view = 45
        boresight = np.dot([0, 0, 1], self.get_pointing(time, mode="dcm"))
        stars = self.star_catalog.get_region(boresight, np.rad2deg(angle)+field_of_view/2)

        # Extract and scale magnitudes
        mag = self.get_photons(stars["magnitude"], delta_t) /  steps

        # Allocate image
        img = np.zeros((self.__settings["resolution"], self.__settings["resolution"]))

        # Integrate star signals
        for step in range(steps):

            # Apply sensor rotation
            dcm = self.get_pointing(time+step_size*step, mode="dcm")
            vis = np.dot(stars["catalog"], dcm)

            # Project stars
            img_x, img_y = self.get_projection(vis)

            # Shift y-axis origin to upper left corner
            img_y = self.__settings["resolution"] - img_y - 1

            # Check for stars in image bounds
            # *** Set buffer > 0 after implementing self.psf_fcn
            resolution = (self.__settings["resolution"], self.__settings["resolution"])
            in_img = imageutils.in_frame(resolution, img_x, img_y, buffer=-0.01)

            # Create image
            # *** This will eventually be replaced by self.psf_fcn
            for idx in in_img:
                xidx = img_x[idx] - np.floor(img_x[idx])
                yidx = img_y[idx] - np.floor(img_y[idx])
                img[int(np.ceil(img_y[idx])), int(np.ceil(img_x[idx]))] += mag[idx]*xidx*yidx
                img[int(np.floor(img_y[idx])), int(np.ceil(img_x[idx]))] += mag[idx]*xidx*(1-yidx)
                img[int(np.ceil(img_y[idx])), int(np.floor(img_x[idx]))] += mag[idx]*(1-xidx)*yidx
                img[int(np.floor(img_y[idx])), int(np.floor(img_x[idx]))] +=                       \
                                                                          mag[idx]*(1-xidx)*(1-yidx)

        # Integrate external object signals
        for object in self.external_objects:
            for step in range(steps):

                # Compute current time
                current_time = time + step_size*step

                # Apply light time correction
                light_time = positionutils.light_time(object.get_position,           \
                                                      self.get_position, current_time)
                current_time -= light_time

                # Compute relative position in camera frame
                vis = object.in_frame_of(self, current_time)

                # If object is colocated with camera, skip iteration
                if np.allclose(vis, 0):
                    continue

                # Project object
                img_x, img_y = self.get_projection(vis)

                # Shift y-axis origin to upper left corner
                img_y = self.__settings["resolution"] - img_y - 1

                # Check if object is in frame
                # *** Set buffer > 0 after implementing self.psf_fcn
                resolution = (self.__settings["resolution"], self.__settings["resolution"])
                in_img = imageutils.in_frame(resolution, img_x, img_y, buffer=-0.01)

                # If object is in image frame, add to image
                if len(in_img) > 0:

                    # Get photon count
                    mag = self.get_photons(object.get_vismag(current_time,                         \
                                                        self.get_position(current_time)), step_size)

                    # Add to image
                    xidx = img_x - np.floor(img_x)
                    yidx = img_y - np.floor(img_y)
                    img[int(np.ceil(img_y)), int(np.ceil(img_x))] += mag*xidx*yidx
                    img[int(np.floor(img_y)), int(np.ceil(img_x))] += mag*xidx*(1-yidx)
                    img[int(np.ceil(img_y)), int(np.floor(img_x))] += mag*(1-xidx)*yidx
                    img[int(np.floor(img_y)), int(np.floor(img_x))] += mag*(1-xidx)*(1-yidx)

        # Return result
        return img

    def snap(self, time, delta_t):
        """
        Create finished image with specified exposure time.

        Parameters
        ----------
        time : float
            Time to begin exposure. Measured in seconds from epoch.
        delta_t : float
            Desired exposure time. Measured in seconds.

        Returns
        -------
        image : ndarray
            Resulting image array. Each pixel contains an integer value.

        Examples
        --------
        >>> cam = StarCam()
        >>> cam.snap(1)
        array([[1427, 1408, 1429, ..., 1381, 1414, 1404],
               [1418, 1370, 1400, ..., 1389, 1395, 1445],
               [1390, 1445, 1323, ..., 1369, 1408, 1417],
               ...,
               [1372, 1469, 1393, ..., 1356, 1468, 1412],
               [1324, 1437, 1496, ..., 1419, 1399, 1360],
               [1412, 1450, 1371, ..., 1376, 1367, 1421]])
        """

        # Integrate photons
        image = self.integrate(time, delta_t)

        # Defocus image
        image = signal.convolve2d(image, self.psf, mode='same')

        # Convert to photoelectrons
        image = self.get_photoelectrons(image)

        # Add noise
        image = self.add_noise(image, delta_t)

        # Saturate
        image = self.get_saturation(image)

        # Return
        return image

    def set_noise_fcn(self, fcn):
        """
        Set internal noise function.

        Parameters
        ----------
        fcn : function
            Input noise function. Output image must be the same size as input.
            See notes for details about the required function format.

        Returns
        -------
        None

        See Also
        --------
        StarCam.set_noise_preset, StarCam.add_noise

        Notes
        -----
        Function must be of the form noisy_image = f(image, delta_t).
        Below are two valid function definition templates.

        def user_fcn(image, delta_t):
            ...
            return noisy_image

        user_fcn = lambda image, delta_t: ...

        Examples
        --------
        >>> cam = StarCam()
        >>> fcn = lambda image, delta_t: image+np.random.rand(*image.shape)
        >>> cam.set_noise_fcn(fcn)
        """

        # Validate input
        if not callable(fcn):
            raise ValueError("Must provide callable function.")
        if fcn(np.zeros(16), 0).shape != (16,):
            raise ValueError("Function output must be the same size as input.")

        # Set function
        self.noise_fcn = fcn

    def set_noise_preset(self, preset, **kwargs):
        """
        Choose preset noise model & assign noise values. Current options are:

        "poisson" -- Poisson-distributed noise.
        "gaussian" -- Gaussian approximation to poisson noise.
        "off" -- Turn image noise off.

        Parameters
        ----------
        preset : str
            Name of chosen preset.
        dark_current : float, optional
            Sensor dark current noise level. Measured in photoelectrons per
            second. Required for "gaussian" and "poisson" presets.
        read_noise : float, optional
            Sensor read noise. Measured in photoelectrons.  Required for
            "gaussian" and "poisson" presets.

        Returns
        -------
        None

        See Also
        --------
        StarCam.set_noise_fcn, StarCam.add_noise

        Notes
        -----
        The default noise for the StarCam object is poisson noise with
        dark_current=1200 and read_noise=200.

        Examples
        --------
        >>> cam = StarCam()
        >>> cam.set_noise_preset("poisson", dark_current=1200, read_noise=200)
        """

        # Poisson model
        if preset.lower() == "poisson":

            # Check input
            if "dark_current" not in kwargs or "read_noise" not in kwargs:
                raise ValueError("Must provide the following keyword arguments for poisson-        \
                                                          type noise: 'dark_current', 'read_noise'")

            # Set function
            noise_fcn = lambda image, delta_t: imageutils.poisson_noise(image, delta_t,            \
                                                       kwargs["dark_current"], kwargs["read_noise"])
            self.set_noise_fcn(noise_fcn)

        # Gaussian model
        elif preset.lower() == "gaussian":
            if "dark_current" not in kwargs or "read_noise" not in kwargs:
                raise ValueError("Must provide the following keyword arguments for poisson-        \
                                                          type noise: 'dark_current', 'read_noise'")

            # Set function
            noise_fcn = lambda image, delta_t: imageutils.gaussian_noise(image, delta_t,           \
                                                       kwargs["dark_current"], kwargs["read_noise"])
            self.set_noise_fcn(noise_fcn)

        elif preset.lower() == "off":

            # Set function
            self.set_noise_fcn(lambda image, delta_t: image)

        # Invalid input
        else:
            raise NotImplementedError("Invalid noise preset. Available options are: poisson,       \
                                                                                    gaussian, off.")

    def add_noise(self, image, delta_t):
        """
        Add noise to image using internal noise model.

        Parameters
        ----------
        image : ndarray
            Input image array. All values should be measured in photoelectrons.
        delta_t : float
            Exposure time in seconds.

        Returns
        -------
        None

        See Also
        --------
        StarCam.set_noise_fcn, StarCam.set_noise_preset

        Examples
        --------
        >>> cam = StarCam()
        >>> cam.set_noise_preset("poisson", dark_current=1200, read_noise=200)
        >>> cam.add_noise(np.zeros((4,4)), 1)
        array([[1398, 1459, 1369, 1466],
               [1375, 1302, 1416, 1465],
               [1370, 1434, 1375, 1463],
               [1491, 1400, 1384, 1381]])
        """

        # Check if noise function is set
        return self.noise_fcn(image, delta_t)

    def set_sensitivity_fcn(self, fcn):
        """
        Set internal conversion between visible magnitudes and photon counts.

        Parameters
        ----------
        fcn : function
            Input photon function. Output must be the same size as input. See
            notes for details about the required function format.

        Returns
        -------
        None

        See Also
        --------
        StarCam.set_sensitivity_preset, StarCam.get_photons

        Notes
        -----
        Function must be of the form photon_count = f(magnitude, delta_t).
        Below are two valid function definition templates.

        def user_fcn(magnitude, delta_t):
            ...
            return photon_count

        user_fcn = lambda magnitude, delta_t: ...

        Examples
        --------
        >>> cam = StarCam()
        >>> fcn = lambda vismags, delta_t: 100*vismags
        >>> cam.set_sensitivity_fcn(fcn)
        """

        # Check for valid input
        if not callable(fcn):
            raise ValueError("Must provide callable function.")

        # Check that input function supports multiple inputs
        if len(fcn([1, 2], 1)) != 2:
            raise ValueError("Input function must support multiple inputs and return an equivalent \
                                                                                 number of values.")

        # Set function
        self.sensitivity_fcn = fcn

    def set_sensitivity_preset(self, preset, **kwargs):
        """
        Choose preset sensitivity model & assign values. This model defines the
        internal conversion between visible magnitudes and photon counts Current
        options are:

        "default" -- A log-scale magnitude to photon conversion.

        Parameters
        ----------
        preset : str
            Name of chosen preset.
        aperture : float, optional
            Aperture area in mm^2. Required for "default" preset.
        mv0_flux : float, optional
            Photoelectrons per second per mm^2 of aperture area. Required for
            "default" preset.

        Returns
        -------
        None

        See Also
        --------
        StarCam.set_sensitivity_fcn, StarCam.get_photons, 
        imageutils.vismag2photon

        Notes
        -----
        The default values for the StarCam object are 1087 mm^2 aperture area
        and 19,000 photons per mm^2 of aperture area per second.

        Examples
        --------
        >>> cam = StarCam()
        >>> cam.set_sensitivity_preset("default", aperture=1087, mv0_flux=19000)
        """

        # Set default option
        if preset.lower() == "default":

            # Check input
            if "aperture" not in kwargs or "mv0_flux" not in kwargs:
                raise ValueError("Must provide the following keyword arguments for this preset:    \
                                                                            'aperture', 'mv0_flux'")

            # Build function & set
            sensitivity_fcn = lambda vismags, delta_t: imageutils.vismag2photon(vismags, delta_t,  \
                                                             kwargs["aperture"], kwargs["mv0_flux"])
            self.set_sensitivity_fcn(sensitivity_fcn)

        # Handle invalid option
        else:
            raise NotImplementedError("Invalid preset option.")

    def get_photons(self, magnitudes, delta_t):
        """
        Convert array of visible magnitudes to photoelectron counts using the
        internally-defined sensitivity model.

        Parameters
        ----------
        magnitudes : ndarray
            Array of visible magnitudes to be converted.
        delta_t : float
            Sensor exposure time in seconds.

        Returns
        -------
        photon_count : ndarray
            Total photon count for each input visible magnitude.

        See Also
        --------
        StarCam.set_sensitivity_fcn, StarCam.set_sensitivity_preset

        Examples
        --------
        >>> cam = StarCam()
        >>> cam.set_sensitivity_preset("default", aperture=1087, mv0_flux=19000)
        >>> cam.get_photons(7, 0.1)
        3383.7875200000003
        """

        # Compute photon count
        return self.sensitivity_fcn(magnitudes, delta_t)

    def set_projection_fcn(self, fcn, resolution):
        """
        Set internal projection model for incoming photons.

        Parameters
        ----------
        fcn : function
            Input projection function. Output must be the same size as input.
            See notes for details about the required function format.
        resolution : int
            Resolution of the sensor.

        Returns
        -------
        None

        See Also
        --------
        StarCam.set_projection_preset, StarCam.get_projection

        Notes
        -----
        Function must be of the form img_x, img_y = f(vectors) where vectors
        is an Nx3 array of unit vectors describing visible objects. Function
        must return image-plane (x,y) coordinate in two separate vectors. Below
        is a valid function definition templates.

        def user_fcn(vectors):
            ...
            return img_x, img_y

        Examples
        --------
        >>> cam = StarCam()
        >>> def proj_fcn(vectors):
        ...     img_x = np.divide(vectors[:, 0], vectors[:, 2])
        ...     img_y = np.divide(vectors[:, 1], vectors[:, 2])
        ...     return img_x, img_y
        ...
        >>> cam.set_projection_fcn(proj_fcn, resolution=1024)
        >>> cam.projection_fcn(np.array([[0, 0, 1]]))
        (array([ 0.]), array([ 0.]))
        """

        # Check for valid resolution
        if resolution <= 0 or not isinstance(resolution, int):
            raise ValueError("Resolution must be integer-valued and positive.")

        # Check for valid function
        if not callable(fcn):
            raise ValueError("Must provide callable function.")

        # Set function
        self.__settings["resolution"] = resolution
        self.projection_fcn = fcn

    def set_projection_preset(self, preset, **kwargs):
        """
        Choose preset projection model & assign values. Current options are:

        "pinhole" -- Pinhole projection model.

        Parameters
        ----------
        preset : str
            Name of chosen preset.
        focal_len : float, optional
            Focal length of the sensor in mm. Required as keyword argument for
            "pinhole" preset.
        pixel_size : float, optional
            Physical pixel size in mm. Pixels are assume square. Required as
            keyword argument for "pinhole" preset.
        resolution : int, optional
            Resolution of the sensor. Default is a square 1024x1024 image.

        Returns
        -------
        None

        See Also
        --------
        StarCam.set_projection_fcn, StarCam.get_projection, 

        Notes
        -----
        The default setting for the StarCam object is the "pinhole" model with
        a focal length of 93 mm, 0.016 mm pixels, and a resolution of 512x512.

        Examples
        --------
        >>> cam = StarCam()
        >>> cam.set_projection_preset("pinhole", focal_len=93, pixel_size=0.016)
        >>> cam.projection_fcn(np.array([[0, 0, 1]]))
        (array([ 512.5]), array([ 512.5]))
        """

        # Set default resolution
        if "resolution" not in kwargs:
            kwargs["resolution"] = 1024

        # Handle pinhole option
        if preset.lower() == "pinhole":

            # Check input
            if "focal_len" not in kwargs or "pixel_size" not in kwargs:
                raise ValueError("Must provide the following keyword arguments for this preset:    \
                                                                         'focal_len', 'pixel_size'")

            # Build function & set
            proj_fcn = lambda vectors: projectionutils.pinhole_project(vectors,                    \
                                    kwargs["focal_len"], kwargs["pixel_size"], kwargs["resolution"])
            self.set_projection_fcn(proj_fcn, kwargs["resolution"])

        # Handle invalid option
        else:
            raise NotImplementedError("Invalid preset option.")

    def get_projection(self, vectors):
        """
        Get projected image-plane coordinates for an input vector using the
        internal projection model.

        Parameters
        ----------
        vectors : ndarray
            Body vectors to be projected into the image plane. Array should be
            Nx3 where N is the number of vectors.

        Returns
        -------
        img_x : ndarray
            Array of x-coordinates (N elements).
        img_y : ndarray
            Array of y-coordinates (N elements).

        See Also
        --------
        StarCam.set_projection_fcn, StarCam.set_projection_preset

        Examples
        --------
        >>> cam = StarCam()
        >>> cam.set_projection_preset("pinhole", focal_len=93, pixel_size=0.016)
        >>> cam.get_projection(np.array([[0, 0, 1]]))
        (array([ 512.5]), array([ 512.5]))
        """

        # Compute projection
        return self.projection_fcn(vectors)

    def set_quantum_efficiency_fcn(self, fcn):
        """
        Set function to simulate CCD quantum efficiency.

        Parameters
        ----------
        fcn : function
            Input quantum efficiency function. Function should convert from
            a continous photon count to a discrete photoelectron count. See
            notes for details about the required function format.

        Returns
        -------
        None

        See Also
        --------
        StarCam.set_quantum_efficiency_preset, StarCam.get_photoelectrons

        Notes
        -----
        Function must be of the form photoelectron_count = f(image).
        Below are two valid function definition templates.

        def user_fcn(image):
            ...
            return photoelectron_count

        user_fcn = lambda image: ...

        Examples
        --------
        >>> cam = StarCam()
        >>> fcn = lambda image: np.floor(image * 0.22)
        >>> cam.set_quantum_efficiency_fcn(fcn)
        """

        # Check function validity
        if not callable(fcn):
            raise ValueError("Must provide callable function.")
        if fcn(np.zeros((16, 32))).shape != (16, 32):
            raise ValueError("Quantum efficiency function output size must be equal to input.")

        # Set function
        self.quantum_efficiency_fcn = fcn

    def set_quantum_efficiency_preset(self, preset, **kwargs):
        """
        Choose preset quantum efficiency model & assign values. Options are:

        "constant" -- Equal quantum efficiency for every pixel.
        "gaussian" -- Gaussian-distributed quantum efficiency values for each
                      pixel.
        "polynomial" -- Quantum efficiency described by a radial polynomial:
                        qe(r) = a_0 + a_1*r + a_2*(r**2) + ... + a_n*(r**n)

        Parameters
        ----------
        preset : str
            Name of chosen preset.
        quantum_efficiency : float, optional
            Relationship between photons and photoelectrons. Measured as the
            number of photoelectrons per photon. Required for "constant" preset.
        sigma : float, optional
            Desired standard deviation of random values. Required for "gaussian"
            preset.
        seed : float, optional
            Random number generator seed. Optional for "gaussian" preset.
        poly : ndarray, optional
            Polynomial coefficient array required for "polynomial" preset.
            Elements designated such that  poly[i,j] * x^i * y^j. The origin of
            the (x,y) pixel coordinate system is the geometric center of the
            image. Coefficient array, poly, must be 2 dimensional.

        Returns
        -------
        None

        See Also
        --------
        StarCam.set_quantum_efficiency_fcn, StarCam.get_photoelectrons

        Notes
        -----
        The StarCam object uses the 'constant' preset by default with a quantum
        efficiency parameter of 0.22.

        Examples
        --------

        Constant QE
        >>> cam = StarCam()
        >>> cam.set_quantum_efficiency_preset("constant", 0.22)

        Polynomial QE with qe(x,y) = 1 + x**2 + y**2
        >>> coeffs = np.array([[1,0,1], [0,0,0], [1,0,0]])
        >>> cam.set_quantum_efficiency_preset("polynomial", poly=coeffs)
        """

        # Set default option
        if preset.lower() == "constant":

            # Check input
            if "quantum_efficiency" not in kwargs:
                raise ValueError("Must provide the following keyword arguments for this preset:    \
                                                                              'quantum_efficiency'")

            # Build function & set
            qe_fcn = lambda image: imageutils.apply_constant_quantum_efficiency(image,            \
                                                                       kwargs["quantum_efficiency"])
            self.set_quantum_efficiency_fcn(qe_fcn)

        # Set gaussian option
        elif preset.lower() == "gaussian":

            # Check input
            if "quantum_efficiency" not in kwargs or "sigma" not in kwargs:
                raise ValueError("Must provide the following keyword arguments for this preset:    \
                                                                     'quantum_efficiency', 'sigma'")

            # Set seed, if necessary
            if "seed" not in kwargs:
                kwargs["seed"] = np.random.rand()

            # Build function & set
            qe_fcn = lambda image: imageutils.apply_gaussian_quantum_efficiency(image,             \
                                      kwargs["quantum_efficiency"], kwargs["sigma"], kwargs["seed"])
            self.set_quantum_efficiency_fcn(qe_fcn)

        # Set polynomial option
        elif preset.lower() == "polynomial":

            # Check input
            if "poly" not in kwargs:
                raise ValueError("Must provide the following keyword arguments for this preset:    \
                                                                                           'poly'.")

            # Build function
            qe_fcn = lambda image: imageutils.apply_polynomial_quantum_efficiency(image,           \
                                                                                     kwargs["poly"])
            self.set_quantum_efficiency_fcn(qe_fcn)

        # Handle invalid option
        else:
            raise NotImplementedError("Invalid preset option.")

    def get_photoelectrons(self, photon_image):
        """
        Get photoelectron count from photon count with internal quantum
        efficiency model.

        Parameters
        ----------
        photon_image : ndarray
            Input image where each pixel contains a total photon count.

        Returns
        -------
        photoelectron_image : ndarray
            Scaled, discrete-valued image where each pixel contains a photo-
            electron count.

        See Also
        --------
        StarCam.set_quantum_efficiency_fcn, StarCam.set_quantum_efficiency_preset

        Examples
        --------
        >>> cam = StarCam()
        >>> cam.set_quantum_efficiency_preset("constant", quantum_efficiency=0.2)
        >>> cam.get_saturation(5*np.ones((4,4)))
        array([[ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.]])
        """

        # Compute photoelectron count
        return self.quantum_efficiency_fcn(photon_image)

    def set_saturation_fcn(self, fcn):
        """
        Set function to simulate sensor-level saturation thresholding.

        Parameters
        ----------
        fcn : function
            Input saturation function. Must be of the form f(image). Output
            must be the same size as input.

        Returns
        -------
        None

        See Also
        --------
        StarCam.set_saturation_preset, StarCam.get_saturation

        Notes
        -----
        Function must be of the form saturated_image = f(image).
        Below are two valid function definition templates.

        def user_fcn(image):
            ...
            return saturated_image

        user_fcn = lambda image: ...

        Examples
        --------
        >>> cam = StarCam()
        >>> fcn = lambda image: np.floor(image)
        >>> cam.set_saturation_fcn(fcn)
        """

        # Check function validity
        if not callable(fcn):
            raise ValueError("Must provide callable function.")
        if fcn(np.zeros((16, 32))).shape != (16, 32):
            raise ValueError("Saturation function output size must be equal to input.")

        # Set function
        self.saturation_fcn = fcn

    def set_saturation_preset(self, preset, **kwargs):
        """
        Choose preset pixel saturation model & assign values. Current options
        are:

        "no_bleed" -- Saturation with no cross-pixel bleed.
        "off"      -- No saturation.

        Parameters
        ----------
        preset : str
            Name of chosen preset.
        bit_depth : int, optional
            Number of bits used to store each pixel. Required for "no_bleed"
            preset. Maximum value for a pixel is 2**bit_depth - 1.

        Returns
        -------
        None

        See Also
        --------
        StarCam.set_saturation_fcn, StarCam.get_saturation

        Notes
        -----
        The StarCam object uses the 'no_bleed' preset by default with a bit
        depth of 16.

        Examples
        --------
        >>> cam = StarCam()
        >>> cam.set_saturation_preset("no_bleed", bit_depth=16)
        """

        # Set default option
        if preset.lower() == "no_bleed":

            # Check input
            if "bit_depth" not in kwargs:
                raise ValueError("Must provide the following keyword arguments for this preset:    \
                                                                                       'bit_depth'")

            # Build function & set
            saturation_fcn = lambda image: imageutils.saturate(image, kwargs["bit_depth"])
            self.set_saturation_fcn(saturation_fcn)

        elif preset.lower() == "off":

            # Set function
            self.set_saturation_fcn(lambda image: image)

        # Handle invalid option
        else:
            raise NotImplementedError("Invalid preset option.")

    def get_saturation(self, image):
        """
        Saturate image input with internal pixel saturation model.

        Parameters
        ----------
        image : ndarray
            Input image where each pixel contains a total photoelectron count.

        Returns
        -------
        saturated_image : ndarray
            Saturated image. Output image is the same size as the input.

        See Also
        --------
        StarCam.set_saturation_fcn, StarCam.set_saturation_preset

        Examples
        --------
        >>> cam = StarCam()
        >>> cam.set_saturation_preset("no_bleed", bit_depth=2)
        >>> cam.get_saturation(16*np.ones((4,4)))
        array([[ 3.,  3.,  3.,  3.],
               [ 3.,  3.,  3.,  3.],
               [ 3.,  3.,  3.,  3.],
               [ 3.,  3.,  3.,  3.]])
        """

        # Saturate image
        return self.saturation_fcn(image)
