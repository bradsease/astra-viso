import numpy as np
from numba import jit

class starcam:
    
    def __init__(self):
        
        # Set default camera parameters
        # **** Should model this after real camera
        # Need to change convention on these variables
        self.f                = 93        # Focal length      (mm)
        self.s                = 0.016     # Pixel size        (mm)
        self.r                = 1024      # Resolution        (px)
        self.aperture         = 1087      # Aperture          (mm^2)
        self.mv0_flux         = 19000     # Mv=0 photon flux  (photons/s/mm^2)
        self.psf              = None
        self.psf_model        = "blur"    # Blur or explicit(not supported, yet)
        self.setpsf(7,1)
        self.projection_model = "pinhole" # Pinhole or polynomial(not supported)
        
        # Set default noise
        self.photon2elec  = 0.22 # photon / e^-
        self.read_noise   = 200  # e^-
        self.dark_current = 1200 # e^- / s
        self.noise_model  = "poisson" # Poisson or Gaussian
        
        # Set default star catalog
        self.stars = None
        self.mags  = None
        self.setcat("test")
        
        # Set default attitude properties
        # Cam.mount
        self.DCM   = np.eye(3)
        self.omega = np.zeros((3,1))
    
    # Set camera parameters
    def set(self, f=None, r=None, fov=None, s=None):

        # Check input arguments
        argnone = (f is None) + (r is None) + (fov is None) + (s is None)
        if (argnone > 1 or argnone == 0):
            print("Incorrect number of arguments for set()! \n"
                  "Must define three variables of (f, res, fov, s).")
            return -1

        # Solve for remaining variable
        if   (f is None):
            f   = s * res / (2 * np.tan(np.deg2rad(fov/2)))
        elif (r is None):
            r   = int(f * (2 * np.tan(np.deg2rad(fov/2))) / s)
        #elif (fov is None):
        #    fov = np.rad2deg(2 * np.arctan(s * r/2/f))
        elif (s is None):
            s   = f * (2 * np.tan(np.deg2rad(fov/2))) / r

        # Set object values
        self.f = f
        self.s = s
        self.r = r
        
    # Set PSF to Gaussian kernel
    # In the future should have a separate function to handle explicit
    # PSF definitions. 
    def setpsf(self, size, sigma):
        
        # Enforce odd dimensions
        if (size % 2 == 0):
            size = size + 1
        
        # Allocate variables
        halfwidth = (size-1)/2
        kernel = np.zeros((size,size))

        # Create kernel
        for row in range(size):
            for col in range(size):
                kernel[row,col] = np.exp(-0.5 * ((row-halfwidth)**2 + (col-halfwidth)**2) / sigma**2)

        # Normalize and return
        self.psf = kernel / np.sum(kernel)
        
    # Choose built-in star catalog by name
    def setcat(self, name):
        self.stars = np.array([[0, 0.004, 0], [0, 0, 0.004], [1, 1, 1]]).T
        self.mags  = np.array([3, 4, 5])
    
    def body2plane(self, v):
        
        # Check input
        if (type(v) is not np.ndarray):
            v = np.array(v)
        if (len(v.shape) == 1):
            v = v.reshape(1,3)
        
        # Project input vectors
        if   (self.projection_model == "pinhole"):
        
            # Pinhole projection equations
            img_x = (self.f/self.s) * np.divide(v[:,0], v[:,2]) + (self.r+1)/2
            img_y = (self.f/self.s) * np.divide(v[:,1], v[:,2]) + (self.r+1)/2
            
        elif (self.projection_model == "polynomial"):
        
            # To be implemented...
            pass
        
        # Return coordinates
        return np.array([img_x,img_y])
        
    def plane2body(self, xy):
        
        # To be implemented...
        pass
    
    # .integrate
    # Snap an image with set integration time
    #
    def integrate(self, dt):
        
        # Rotate star catalog
        vis     = np.dot(self.stars, self.DCM)
        visinds = [i for i in range(len(vis[:,-1])) if vis[i,-1] > 0]
        vis     = vis[:, visinds]
        
        # Extract and scale magnitudes
        mags    = self.mv0_flux * (1 / (2.5**self.mags[visinds])) * dt * self.aperture
        
        # Project remaining stars
        img_x   = (self.f/self.s) * np.divide(vis[:,0], vis[:,2]) + (self.r+1)/2
        img_y   = (self.f/self.s) * np.divide(vis[:,1], vis[:,2]) + (self.r+1)/2
        
        # Create image
        img     = np.zeros((self.r, self.r))
        in_img  = [i for i in range(len(img_x)) if (img_x[i] > 0        and  
                                                    img_x[i] < self.r-1 and
                                                    img_y[i] > 0        and
                                                    img_y[i] < self.r-1)]
        for k in range(len(in_img)):
            i  = in_img[k]
            xd = img_x[i] - np.floor(img_x[i])
            yd = img_y[i] - np.floor(img_y[i])
            img[int(np.ceil(img_y[i])) , int(np.ceil(img_x[i])) ] += mags[i] *    xd  *    yd
            img[int(np.floor(img_y[i])), int(np.ceil(img_x[i])) ] += mags[i] *    xd  * (1-yd)
            img[int(np.ceil(img_y[i])) , int(np.floor(img_x[i]))] += mags[i] * (1-xd) *    yd
            img[int(np.floor(img_y[i])), int(np.floor(img_x[i]))] += mags[i] * (1-xd) * (1-yd)
        
        return img
    
    # Create finished image
    def snap(self, dt):
        
        # Integrate photons
        image = self.integrate(dt)
        
        # Defocus image
        image = self.defocus(image, self.psf)
        
        # Convert to photoelectrons
        image = np.floor(image * self.photon2elec)
        
        # Add noise
        image = self.addnoise(image, dt)
        
        # Return
        return image
        
    # Blur image 
    @jit
    def defocus(self, img_in, psf):

        # Allocate variables
        s       = psf.shape[0]
        s_half  = int(np.floor(psf.shape[0]/2))
        m,n     = img_in.shape
        img     = np.copy(img_in)
        img_pad = np.zeros((m+2*s_half, n+2*s_half))
        img_pad[s_half:-(s_half), s_half:-(s_half)] = img

        # Convolve image with kernel
        for i in range(m):
            for j in range(n):
                img[i,j] = np.sum(img_pad[0+i:s+i, 0+j:s+j] * psf)

        # Return result
        return img
    
    # Add noise to image
    def addnoise(self, image, dt):

        # Poisson model
        if   (self.noise_model.lower() == "poisson"):

            # Add shot noise
            image  = np.random.poisson(image)

            # Add dark current
            image += np.random.poisson(self.dark_current*dt, image.shape)

            # Add read noise
            image += np.random.poisson(self.read_noise, image.shape)

        # Gaussian approximate model
        elif (self.noise_model.lower() == "gaussian"):

            # Add shot noise
            image += np.round(np.sqrt(image) * np.random.randn(*image.shape))

            # Add dark current
            image += np.round(self.dark_current*dt + np.sqrt(self.dark_current*dt) * np.random.randn(*image.shape))

            # Add read noise
            image += np.round(self.read_noise + np.sqrt(self.read_noise) * np.random.randn(*image.shape))

        # Return
        return image