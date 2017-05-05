"""
Astra-Viso star camera module.
"""
import numpy as np
from astraviso import worldobject
from astraviso import starmap
from astraviso import imageutils

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

        # Set default camera parameters
        # **** Should model this after real camera
        # Need to change convention on these variables
        self.focal_len = 93               # Focal length      (mm)
        self.pixel_size = 0.016           # Pixel size        (mm)
        self.resolution = 1024            # Resolution        (px)
        self.setpsf(7, 1)                 # To be removed...

        # Internal function variables
        self.sensitivity_fcn = None
        self.projection_fcn = None
        self.quantum_efficiency_fcn = None
        self.noise_fcn = None
        self.saturation_fcn = None

        # Set star catalog defaults
        self.star_catalog = starmap.StarMap()
        self.star_catalog.load_preset("random", 1000)

        # Set sensor pointing default
        worldobject.WorldObject.__init__(self)
        self.set_pointing_preset("kinematic", np.array([0, 0, 0, 1, 0, 0, 0]))

        # Set CCD defaults
        self.set_saturation_preset("no_bleed", bit_depth=16)
        self.set_quantum_efficiency_preset("constant", quantum_efficiency=0.22)
        self.set_noise_preset("poisson", dark_current=1200, read_noise=200)
        self.set_sensitivity_preset("default", aperture=1087, mv0_flux=19000)

        # Internal settings
        self.__settings = {}
        self.__settings["max_angle_step"] = 1e-4
        self.__settings["projection_model"] = "pinhole" # To be removed...

        # External objects
        self.external_objects = []

    def set(self, focal_len=None, resolution=None, fov=None, pixel_size=None):
        """
        Set camera parameters.
        """

        # Check input arguments
        argnone = (focal_len is None) + (resolution is None) + (fov is None) + (pixel_size is None)
        if argnone > 1 or argnone == 0:
            print("Incorrect number of arguments for set()! \n"
                  "Must define three variables of (f, res, fov, s).")
            return -1

        # Solve for remaining variable
        if focal_len is None:
            focal_len = pixel_size * resolution / (2 * np.tan(np.deg2rad(fov/2)))
        elif resolution is None:
            resolution = int(focal_len * (2 * np.tan(np.deg2rad(fov/2))) / pixel_size)
        elif pixel_size is None:
            pixel_size = focal_len * (2 * np.tan(np.deg2rad(fov/2))) / resolution

        # Set object values
        self.focal_len = focal_len
        self.pixel_size = pixel_size
        self.resolution = resolution

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
        index: int
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

    def body2plane(self, vectors):
        """
        Convert body-fixed position vector to image-plane coordinates.
        """

        # Check input
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, 3)

        # Project input vectors
        if self.__settings["projection_model"] == "pinhole":

            # Pinhole projection equations
            f_over_s = (self.focal_len/self.pixel_size)
            half_res = (self.resolution+1)/2
            img_x = f_over_s * np.divide(vectors[:, 0], vectors[:, 2]) + half_res
            img_y = f_over_s * np.divide(vectors[:, 1], vectors[:, 2]) + half_res

        elif self.__settings["projection_model"] == "polynomial":

            # To be implemented...
            pass

        # Return coordinates
        return img_x, img_y

    def integrate(self, delta_t):
        """
        Compute pixel values after set exposure time.
        """

        # Determine step size
        angle = np.arccos(0.5*(np.trace(np.dot(self.get_pointing(0, mode="dcm"),                   \
                                                      self.get_pointing(delta_t, mode="dcm").T))-1))
        steps = int(np.ceil(max(1.0 + angle/self.__settings["max_angle_step"], 1.0)))
        step_size = delta_t / steps

        # Allocate image
        img = np.zeros((self.resolution, self.resolution))

        # Extract subset of stars from catalog
        field_of_view = np.rad2deg(2*np.arctan(self.pixel_size*self.resolution/2/self.focal_len))
        boresight = np.dot([0, 0, 1], self.get_pointing(0, mode="dcm"))
        stars = self.star_catalog.getregion(boresight, np.rad2deg(angle)+field_of_view/2)

        # Extract and scale magnitudes
        mag = self.get_photons(stars["magnitude"], delta_t) /  steps

        # Integrate star signals
        for step in range(steps):

            # Rotate stars
            dcm = self.get_pointing(step_size*step, mode="dcm")
            vis = np.dot(stars["catalog"], dcm)

            # Project stars
            img_x, img_y = self.body2plane(vis)

            # Check for stars in image bounds
            in_img = [idx for idx in range(len(img_x)) if (img_x[idx] > 0                 and
                                                           img_x[idx] < self.resolution-1 and
                                                           img_y[idx] > 0                 and
                                                           img_y[idx] < self.resolution-1)]

            # Create image
            for idx in in_img:
                xidx = img_x[idx] - np.floor(img_x[idx])
                yidx = img_y[idx] - np.floor(img_y[idx])
                img[int(np.ceil(img_y[idx])), int(np.ceil(img_x[idx]))] += mag[idx]*xidx*yidx
                img[int(np.floor(img_y[idx])), int(np.ceil(img_x[idx]))] += mag[idx]*xidx*(1-yidx)
                img[int(np.ceil(img_y[idx])), int(np.floor(img_x[idx]))] += mag[idx]*(1-xidx)*yidx
                img[int(np.floor(img_y[idx])), int(np.floor(img_x[idx]))] +=                       \
                                                                          mag[idx]*(1-xidx)*(1-yidx)

        return img

    # Create finished image
    def snap(self, delta_t):
        """
        Create finished image with specified exposure time.
        """

        # Integrate photons
        image = self.integrate(delta_t)

        # Defocus image
        image = imageutils.conv2(image, self.psf)

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
                                                                                         gaussian.")

    def add_noise(self, image, delta_t):
        """
        Add noise to image.
        """

        if self.noise_fcn is not None:
            return self.noise_fcn(image, delta_t)
        else:
            return image

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

    def set_projection_fcn(self):
        """
        """

        # To be implemented...
        raise NotImplementedError("Not yet implemented!")

    def set_projection_preset(self):
        """
        """

        # To be implemented...
        raise NotImplementedError("Not yet implemented!")

    def get_projection(self, magnitudes):
        """
        """

        # To be implemented...
        raise NotImplementedError("Not yet implemented!")

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
            raise ValueError("Saturation function output size must be equal to input.")

        # Set function
        self.quantum_efficiency_fcn = fcn

    def set_quantum_efficiency_preset(self, preset, **kwargs):
        """
        Choose preset quantum efficiency model & assign values. Current options are:

        "constant" -- Equal quantum efficiency for every pixel.
        "gaussian" -- Gaussian-distributed quantum efficiency values for each
                      pixel.

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
        >>> cam = StarCam()
        >>> cam.set_quantum_efficiency_preset("constant", 0.22)
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
