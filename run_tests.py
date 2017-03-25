import os
import sys

# Test header function
def print_header(segment):
    print("")
    print("----------------------------------------------------------------------")
    print("Running " + segment + " unit tests...")
    print("----------------------------------------------------------------------")

# Check input
available_tests = ["starcam", "starmap", "worldobject", "pointingutils"]
if (len(sys.argv) > 1):
    tests = sys.argv[1:]
else:
    tests = available_tests
    
# Run tests
for test in tests:
    if (test in available_tests):
        print_header(test)
        command = "python -m unittest unit_tests/" + test + "_test.py -v"
        os.system(command)