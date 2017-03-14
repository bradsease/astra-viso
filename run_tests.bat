@ECHO OFF
echo.
echo ----------------------------------------------------------------------
echo Running starcam unit tests...
echo ----------------------------------------------------------------------
echo.
python -m unittest unit_tests/starcam_test.py -v

echo.
echo ----------------------------------------------------------------------
echo Running starmap unit tests...
echo ----------------------------------------------------------------------
echo.
python -m unittest unit_tests/starmap_test.py -v

echo.
echo ----------------------------------------------------------------------
echo Running worldobject unit tests...
echo ----------------------------------------------------------------------
echo.
python -m unittest unit_tests/worldobject_test.py -v

echo.
echo ----------------------------------------------------------------------
echo Running pointing-util unit tests...
echo ----------------------------------------------------------------------
echo.
python -m unittest unit_tests/pointingutils_test.py -v
::python -m unittest unit_tests/worldobject_test.py -v
::python -m unittest discover unit_tests "*_test.py" -v

echo.