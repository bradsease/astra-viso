# Astra Viso

A python simulator for star and unresolved spacecraft imaging.


Astra Viso is a full-featured toolset for simulating stars and unresolved space objects. With this software it is simple to set up a simulated sensor and begin generating imagery immediately. The tools allow the user to switch between a number of commonly-used internal modeling presets easily while also having the flexibility to override internal modeling with externally-defined functions.


## Installation

The easiest way to install Astra Viso is with pip. Simply type the following at the command line
```python
pip install astraviso
```
You may also download the package from github as a zip. Extract the zip into a folder and run
```python
python setup.py install
```
Alternatively, you may copy the astraviso folder into the same folder as your other scripts and import it locally.


## Testing

Regardless of the method of installation, it is possible to verify the installation by running internal tests at any time with the following code. This verification routine will run all of the internal tests and may take some time to complete.
```python
import astraviso
astraviso.verify()
```


## Use

Astra Viso contains some simple demo scripts to highlight the capabilities of the package. For information on available demos, check `help(astraviso.run_demo)`. To run the default demo, type
```python
import astraviso
astraviso.run_demo()
```
Astra Viso is broken into 3 primary classes: StarCam, WorldObject, and StarMap. Each of these classes handles a portion of your interaction with the code. The WorldObject class describes the behavior of an object in inertial space. Any space object of interest can be a WorldObject. The StarMap class is a simple interface to manage star catalogs. Ultimately, the StarCam class manages simulating the physics of a CCD sensor and creating an image. The most simple Astra Viso program is
```python
import astraviso as av

# Create star camera instance
cam = av.StarCam()

# Capture image starting at t=0 with an exposure time of 1 second
image = cam.snap(0, 1)

# Display image using the internal tools
av.imageutils.imshow(image)
```
This code creates a default camera. The default star catalog is a randomly-generated set with 10,000 elements.


For additional information, see the full documentation: bradsease.github.io/astraviso

## Requires
* Python 2.7, 3.3+
* numpy
* scipy
* matplotlib
* numba


## Citing Astra Viso

If you use Astra Viso for research, please cite me! Here's a suggested format:

B. Sease, *Astra Viso*, ver. 0.1.0, 2017.
