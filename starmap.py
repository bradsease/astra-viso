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

    def __init__(self, name=None):
        """
        Initialize star map.
        """

        # Stars
        self.catalog = None
        self.magnitude = None
        self.size = 0

        # Load catalog
        if name:
            self.loadpreset(name)
        else:
            self.loadpreset("singlecenter")

    def loadpreset(self, name, *arg):
        """
        Load preset star catalog.
        """

        # Single star on boresight
        if name.lower() == "singlecenter":
            self.catalog = np.array([[0, 0, 1]])
            self.magnitude = np.array([-1])

        # Six stars, one on each axis
        elif name.lower() == "sixfaces":
            self.catalog = np.array([[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0],      \
                                                                                        [-1, 0, 0]])
            self.magnitude = np.array([12, 8, 4, 0, -4, -8])

        # Generate a random catalog
        elif name[0:6].lower() == "random":
            self.catalog = np.zeros((arg[0], 3))
            self.magnitude = 8 + 2*np.random.randn(arg[0])
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
                infile = open("catalogs/" + name.lower() + ".dat", 'rb')
                catalog_file = pickle.load(infile)
                infile.close()

                # Set catalog
                self.catalog = catalog_file["catalog"]
                self.magnitude = catalog_file["magnitude"]
            except ValueError:
                print("Unknown preset: %s" % name)

        # Set size variable
        self.size = len(self.catalog)

    def getregion(self, vector, angle):
        """
        Extract catalog region with direction vector and angle.
        """

        # Enforce normalization of input vector
        if np.linalg.norm(vector) == 0:
            raise ValueError("Central vector must be non-zero.")
        vector = vector / np.linalg.norm(vector)

        # Extract region
        infield = [i for i in range(self.size) if                                                  \
                                 np.arccos(np.dot(vector, self.catalog[i, :])) <= np.deg2rad(angle)]

        return {"catalog"   : self.catalog[infield],
                "magnitude" : self.magnitude[infield]}

    def selectbrighter(self, limit):
        """
        Select only stars brighter than a given magnitude.
        """

        self.downselect(lambda x: x < limit, "magnitude")

    def selectdimmer(self, limit):
        """
        Select only stars dimmer than a given magnitude.
        """

        self.downselect(lambda x: x > limit, "magnitude")

    def selectrange(self, dimmest, brightest):
        """
        Select only stars within a range of magnitudes.
        """

        self.downselect(lambda x: x <= brightest and x >= dimmest, "magnitude")

    def downsample(self, factor, mode="random"):
        """
        Downsample current catalog.
        """

        # Downsample randomly
        if mode.lower() == "random":
            self.downselect(lambda x, idx: np.random.rand() <= 1/factor, "magnitude")

        # Sample at interval
        if mode.lower() == "interval":
            self.downselect(lambda x, idx: np.isclose(idx % factor, 0), "magnitude")

    def downselect(self, func, mode):
        """
        Downselect current catalog according to input function.
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

    def viewfield(self):
        """
        Plot entire catalog.
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
        Plot region of catalog.
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

    def checkvalidity(self):
        """
        Check validity of current catalog.
        """

        # check for normed vectors, fix if not...
        # check for types?
        pass
