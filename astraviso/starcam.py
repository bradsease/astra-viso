"""
Astra-Viso star camera module.
"""
import numpy as np
from numba import jit
from astraviso import worldobject
from astraviso import starmap

class StarCam(worldobject.WorldObject):
    """
    Star camera class.
    """

    def __init__(self):
        """
        StarCam initialization.
        """

        # Set default camera parameters
        # **** Should model this after real camera
        # Need to change convention on these variables
        self.focal_len = 93               # Focal length      (mm)
        self.pixel_size = 0.016           # Pixel size        (mm)
        self.resolution = 1024            # Resolution        (px)
        self.aperture = 1087              # Aperture          (mm^2)
        self.mv0_flux = 19000             # Mv=0 photon flux  (photons/s/mm^2)
        self.psf_model = "blur"           # Blur or explicit(not supported, yet)
        self.setpsf(7, 1)
        self.projection_model = "pinhole" # Pinhole or polynomial(not supported)

        # Set default noise
        self.photon2elec = 0.22           # photon / e^-
        self.read_noise = 200             # e^-
        self.dark_current = 1200          # e^- / s
        self.noise_model = "poisson"      # Poisson or Gaussian

        # Set default star catalog
        self.star_catalog = starmap.StarMap()
        self.star_catalog.loadpreset("random", 50000)

        # Internal settings
        self.max_angle_step = 1e-4

        # Set default attitude properties
        worldobject.WorldObject.__init__(self)
        self.set_pointing_preset("kinematic", np.array([0, 0, 0, 1, 0, 0, 0]))

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

    def add_worldobject(self, object=None):
        """
        Add new or existing world object to catalog.
        """
        raise NotImplementedError("Not yet implemented!")

    def body2plane(self, vectors):
        """
        Convert body-fixed position vector to image-plane coordinates.
        """

        # Check input
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, 3)

        # Project input vectors
        if self.projection_model == "pinhole":

            # Pinhole projection equations
            f_over_s = (self.focal_len/self.pixel_size)
            half_res = (self.resolution+1)/2
            img_x = f_over_s * np.divide(vectors[:, 0], vectors[:, 2]) + half_res
            img_y = f_over_s * np.divide(vectors[:, 1], vectors[:, 2]) + half_res

        elif self.projection_model == "polynomial":

            # To be implemented...
            pass

        # Return coordinates
        return img_x, img_y

    def plane2body(self, image_coord):
        """
        Convert image-plane coordinates to body fixed unit vector.
        """

        # To be implemented...
        pass

    def mag2photon(self, magnitudes):
        """
        Convert vector of visible magnitudes to photoelectrons/second.
        """

        # To be implemented...
        raise NotImplementedError("Not yet implemented!")

    def integrate(self, delta_t):
        """
        Compute pixel values after set exposure time.
        """

        # Determine step size
        angle = np.arccos(0.5*(np.trace(np.dot(self.get_pointing(0, mode="dcm"),                   \
                                                      self.get_pointing(delta_t, mode="dcm").T))-1))
        steps = int(np.ceil(max(1.0 + angle/self.max_angle_step, 1.0)))
        step_size = delta_t / steps

        # Allocate image
        img = np.zeros((self.resolution, self.resolution))

        # Extract subset of stars from catalog
        field_of_view = np.rad2deg(2*np.arctan(self.pixel_size*self.resolution/2/self.focal_len))
        boresight = np.dot([0, 0, 1], self.get_pointing(0, mode="dcm"))
        stars = self.star_catalog.getregion(boresight, np.rad2deg(angle)+field_of_view/2)

        # Extract and scale magnitudes
        mag = self.mv0_flux*(1/(2.5**stars["magnitude"]))*delta_t*self.aperture/steps

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
        image = self.defocus(image, self.psf)

        # Convert to photoelectrons
        image = np.floor(image * self.photon2elec)

        # Add noise
        image = self.addnoise(image, delta_t)

        # Return
        return image

    @jit
    def defocus(self, img_in, psf):
        """
        Defocus image.
        """

        # Allocate variables
        size = psf.shape[0]
        size_half = int(np.floor(psf.shape[0]/2))
        rows, cols = img_in.shape
        img = np.copy(img_in)
        img_pad = np.zeros((rows+2*size_half, cols+2*size_half))
        img_pad[size_half:-(size_half), size_half:-(size_half)] = img

        # Convolve image with kernel
        for row in range(rows):
            for col in range(cols):
                img[row, col] = np.sum(img_pad[row:size+row, col:size+col]*psf)

        # Return result
        return img

    def addnoise(self, image, delta_t):
        """
        Add noise to image.
        """

        # Poisson model
        if self.noise_model.lower() == "poisson":

            # Add shot noise
            image = np.random.poisson(image)

            # Add dark current
            image += np.random.poisson(self.dark_current*delta_t, image.shape)

            # Add read noise
            image += np.random.poisson(self.read_noise, image.shape)

        # Gaussian approximate model
        elif self.noise_model.lower() == "gaussian":

            # Add shot noise
            image += np.round(np.sqrt(image) * np.random.randn(*image.shape))

            # Add dark current
            image += np.round(self.dark_current*delta_t + np.sqrt(self.dark_current*delta_t) *    \
                                                                     np.random.randn(*image.shape))

            # Add read noise
            image += np.round(self.read_noise + np.sqrt(self.read_noise) *                        \
                                                                     np.random.randn(*image.shape))

        # Return
        return image
