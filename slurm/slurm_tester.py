import sys
import random
from pathlib import Path 
from topsim.utils.experiment import Experiment # Test topsim modules can be imported


fname = sys.argv[1]
tid = sys.argv[2]

with open(f"output_{tid}", "w") as fp:
    fp.write(fname)