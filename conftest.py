import pathlib
import sys

# Make the synthetic-data generator in examples/ importable from the tests.
sys.path.insert(0, str(pathlib.Path(__file__).parent / "examples"))
