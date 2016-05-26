import sys
from . import run
result = run()
sys.exit(not result.wasSuccessful())
