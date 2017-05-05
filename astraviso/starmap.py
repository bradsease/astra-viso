"""
Astra-Viso star map module.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class StarMap:
    """
    Star map class.
    """

    def __init__(self, preset=None):
        """
        Initialize new star catalog object. Option to load one of several
        pre-defined catalogs. Options are:

        "singlecenter" -- A single bright star aligned with the z-axis.
        "sixfaces"     -- Six bright stars oriented along each positive and
                          negative axis.
        "random"       -- A randomly generated catalog with a user-defined
                          number of stars.
        "hipparcos"    -- The Hipparcos star catalog. 117,955 total stars [1].
        "tycho"        -- The Tycho-2 star catalog. 1,055,115 total stars [2].

        Parameters
        ----------
        preset : str, optional
            Name of preset star catalog to load.

        Returns
        -------
        starmap : StarMap
            Initialized star catalog object.
        """

        # Stars
        self.catalog = None
        self.magnitude = None
        self.size = 0

        # Load catalog
        if preset:
            self.load_preset(preset)
        else:
            self.load_preset("singlecenter")

    def load_preset(self, preset, *arg):
        """
        Load a preset star catalog. Current available options are:
        
        "singlecenter" -- A single bright star aligned with the z-axis.
        "sixfaces"     -- Six bright stars oriented along each positive and
                          negative axis.
        "random"       -- A randomly generated catalog with a user-defined
                          number of stars.
        "hipparcos"    -- The Hipparcos star catalog. 117,955 total stars [1].
        "tycho"        -- The Tycho-2 star catalog. 1,055,115 total stars [2].

        Parameters
        ----------
        preset : str
            Desired preset option.
        star_count : int, optional
            Number of stars desired from the "random" preset. Required for the
            "random" preset.

        Returns
        -------
        None

        Notes
        -----
        [1] Perryman, Michael AC, et al. "The HIPPARCOS catalogue." Astronomy
            and Astrophysics 323 (1997).
        [2] Høg, Erik, et al. "The Tycho-2 catalogue of the 2.5 million
            brightest stars." Astronomy and Astrophysics 355 (2000): L27-L30.

        Examples
        --------
        >>> catalog = StarMap("tycho")
        >>> catalog.size
        1055115
        >>> catalog.load_preset("hipparcos")
        >>> catalog.size
        117955
        """

        # Single star on boresight
        if preset.lower() == "singlecenter":
            self.catalog = np.array([[0, 0, 1]])
            self.magnitude = np.array([-1])

        # Six stars, one on each axis
        elif preset.lower() == "sixfaces":
            self.catalog = np.array([[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0],      \
                                                                                        [-1, 0, 0]])
            self.magnitude = np.array([12, 8, 4, 0, -4, -8])

        # Generate a random catalog
        elif preset[0:6].lower() == "random":

            # Pre-allocate catalog
            self.catalog = np.zeros((arg[0], 3))
            self.magnitude = 8 + 2*np.random.randn(arg[0])

            # Generate random unit vectors
            for i in range(len(self.catalog)):
                theta = np.arccos(1 - 2 * np.random.rand())
                phi = 2 * np.pi * np.random.rand()
                self.catalog[i] = [np.sin(theta) * np.cos(phi),
                                   np.sin(theta) * np.sin(phi),
                                   np.cos(theta)]

        # Handle any other option
        else:

            # Open pickle file, if it exists
            try:

                # Load file
                infile = open("catalogs/" + preset.lower() + ".dat", 'rb')
                catalog_file = pickle.load(infile)
                infile.close()

                # Set catalog
                self.catalog = catalog_file["catalog"]
                self.magnitude = catalog_file["magnitude"]

            except FileNotFoundError:
                print("Unknown preset: %s" % preset)

        # Set size variable
        self.size = len(self.catalog)

    def get_all(self):
        """
        Export all catalog elements to a dict.

        Parameters
        ----------
        None

        Returns
        -------
        map : dict
            Dictionary containing the star catalog in the form of an Nx3 array
            of unit vectors (key:"catalog") and an array of corresponding
            visible magnitudes (key:"magnitude").

        Examples
        --------
        >>> catalog = StarMap("singlecenter")
        >>> map = catalog.get_all()
        >>> map
        {'catalog': array([[0, 0, 1]]), 'magnitude': array([-1])}
        """

        return {"catalog"   : self.catalog,
                "magnitude" : self.magnitude}

    def get_region(self, vector, angle):
        """
        Extract catalog elements falling within a given angle of a specified
        unit vector.

        Parameters
        ----------
        vector : ndarray
            Three-element array containing a desired unit vector direction.
        angle : float
            Angle about the designated unit vector to accept stars. Measured in
            degrees.

        Returns
        -------
        map : dict
            Dictionary containing the star catalog region in the form of an Nx3
            array of unit vectors (key:"catalog") and an array of corresponding
            visible magnitudes (key:"magnitude").

        Examples
        --------
        >>> catalog = StarMap("hipparcos")
        >>> map = catalog.get_region(np.array([0, 0, 1]), 0.001)
        >>> map
        {'catalog': array([[ -1.68386803e-06,   4.35351891e-06,   1.00000000e+00],
        [ -1.83452395e-06,  -3.16303724e-06,   1.00000000e+00],
        [  1.51683717e-05,   4.10971724e-06,   1.00000000e+00]]), 
        'magnitude': array([ 9.03,  9.02,  8.69])}
        """

        # Enforce normalization of input vector
        if np.linalg.norm(vector) == 0:
            raise ValueError("Central vector must be non-zero.")
        vector = vector / np.linalg.norm(vector)

        # Extract region
        infield = [i for i in range(self.size) if                                                  \
                                 np.arccos(np.dot(vector, self.catalog[i, :])) <= np.deg2rad(angle)]

        # Return result
        return {"catalog"   : self.catalog[infield],
                "magnitude" : self.magnitude[infield]}

    def downselect(self, func, mode):
        """
        Downselect current catalog according to a boolean-valued input function.
        Culls the internal catalog.

        Parameters
        ----------
        func : function
            Boolean-valued selection function. Must accept two inputs. See notes
            for more information on the required input format.
        mode : str
            Target values for downselect operation. Options are "magnitude" or
            "catalog".

        Returns
        -------
        None

        Notes
        -----
        For the "magnitude" mode option, the input function must be of the form:
                               bool = f(magnitude, index)
        where the magnitude value is a scalar float and the index is a scalar
        int. The index value corresponds to the index of the current element.

        For the "catalog" mode option, the input function must be of the form:
                               bool = f(vector, index)
        where the vector value is a 3-element array and the index is a scalar
        int. The index value corresponds to the index of the current element.

        Examples
        --------
        >>> catalog = StarMap("hipparcos")
        >>> catalog.size
        117955
        >>> select_fcn = lambda mag, idx: mag < 6 & idx < 100000
        >>> catalog.downselect(select_fcn, "magnitude")
        >>> catalog.size
        1413
        """

        # Check function input arguments
        if func.__code__.co_argcount == 1:
            fcn = lambda val, idx: func(val)
        elif func.__code__.co_argcount == 2:
            fcn = func
        else:
            print("Improper number of input arguments!")
            return

        # Downselect based on star magnitudes
        if mode.lower() == "magnitude":
            selected = [idx for idx in range(self.size) if fcn(self.magnitude[idx], idx)]

        # Downselect based on star unit vectors
        elif mode.lower() == "catalog":
            selected = [idx for idx in range(self.size) if fcn(self.catalog[idx], idx)]

        # Unsupported option
        else:
            print("Unsupported option: %s" % mode)

        # Finalize downselect
        self.catalog = self.catalog[selected]
        self.magnitude = self.magnitude[selected]
        self.size = len(self.catalog)

    def downsample(self, factor, mode="random"):
        """
        Downsample current catalog.

        Parameters
        ----------
        factor : float
            Factor to downsample by. Resulting catalog length will be
            approximately 1/factor.
        mode : str, optional
            Downsampling mode. Options are "random" or "interval". Default is
            "random". 

        Returns
        -------
        None

        Examples
        --------
        >>> catalog = StarMap("hipparcos")
        >>> catalog.size
        117955
        >>> catalog.downsample(10, mode="interval")
        >>> catalog.size
        11796
        """

        # Check input
        if factor <= 0:
            return

        # Downsample randomly
        if mode.lower() == "random":
            self.downselect(lambda x, idx: np.random.rand() <= 1/factor, "magnitude")

        # Sample at interval
        elif mode.lower() == "interval":
            self.downselect(lambda x, idx: np.isclose(idx % factor, 0), "magnitude")

        # Handle invalid mode
        else:
            raise ValueError("Invalid mode type. Options are: 'random' and 'interval'.")

    def select_brighter(self, limit):
        """
        Select only stars brighter than a given magnitude. Culls the internal
        catalog only.

        Parameters
        ----------
        limit : float
            Visible magnitude limit. Stars with magnitude values less than this
            limit will be selected.

        Returns
        -------
        None

        Examples
        --------
        >>> catalog = StarMap("hipparcos")
        >>> catalog.size
        117955
        >>> catalog.select_brighter(8)
        >>> catalog.size
        41057
        """

        self.downselect(lambda x: x < limit, "magnitude")

    def select_dimmer(self, limit):
        """
        Select only stars dimmer than a given magnitude. Culls the internal
        catalog only.

        Parameters
        ----------
        limit : float
            Visible magnitude limit. Stars with magnitude values greater than
            this limit will be selected.

        Returns
        -------
        None

        Examples
        --------
        >>> catalog = StarMap("hipparcos")
        >>> catalog.size
        117955
        >>> catalog.select_dimmer(8)
        >>> catalog.size
        76561
        """

        self.downselect(lambda x: x > limit, "magnitude")

    def select_range(self, brightest, dimmest):
        """
        Select only stars within a range of magnitudes. Culls the internal
        catalog only.

        Parameters
        ----------
        brightest : float
            Upper limit on star brightness. Stars with magnitude values greater
            than this limit will be selected.
        dimmest : float
            Lower limit on star brightness. Stars with magnitude values less
            than this limit will be selected.

        Returns
        -------
        None

        Examples
        --------
        >>> catalog = StarMap("hipparcos")
        >>> catalog.size
        117955
        >>> catalog.select_dimmer(8)
        >>> catalog.size
        41393
        """

        self.downselect(lambda x: x >= brightest and x <= dimmest, "magnitude")

    def viewfield(self):
        """
        Create 3D plot of the entire catalog.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # Plot data
        fig = plt.figure()
        axis = Axes3D(fig)
        axis.scatter(self.catalog[:, 0], self.catalog[:, 1], self.catalog[:, 2], marker=".",     \
                                                                               color="black", s=3)

        # Show plot
        axis.set_xlim([-1, 1])
        axis.set_ylim([-1, 1])
        axis.set_zlim([-1, 1])
        plt.show()

    def viewregion(self, vector, angle):
        """
        Create 3D plot of a region of the catalog.

        Parameters
        ----------
        vector : ndarray
            Three-element array containing a desired unit vector direction.
        angle : float
            Angle about the designated unit vector to accept stars. Measured in
            degrees.

        Returns
        -------
        None
        """

        # Select region
        region = self.getregion(vector, angle)

        # Plot data
        fig = plt.figure()
        axis = Axes3D(fig)
        axis.scatter(region["catalog"][:, 0], region["catalog"][:, 1], region["catalog"][:, 2],    \
                                                                     marker=".", color="black", s=2)

        # Plot input vector
        axis.quiver(0, 0, 0, vector[0], vector[1], vector[2], color="red", linewidth=1.5)

        # Show plot
        axis.set_xlim([-1, 1])
        axis.set_ylim([-1, 1])
        axis.set_zlim([-1, 1])
        plt.show()
