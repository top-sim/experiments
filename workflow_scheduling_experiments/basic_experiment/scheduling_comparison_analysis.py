
import sys
import numpy as np
import pandas
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path


# Setup all the visualisation nicities
from matplotlib import rcParams

rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = "computer modern roman"
rcParams['font.size'] = 12.0

rcParams['axes.linewidth'] = 1

# X-axis
rcParams['xtick.direction'] = 'in'
rcParams['xtick.minor.visible'] = True
# Y-axis
rcParams['ytick.direction'] = 'in'
rcParams['ytick.minor.visible'] = True

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    RESULT_FILE = sys.argv[1]
