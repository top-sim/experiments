import sys
import numpy as np
import pandas
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

from skaworkflows.common import SI

# Setup all the visualisation nicities
from matplotlib import rcParams

rcParams["text.usetex"] = True
rcParams["font.family"] = "serif"
rcParams["font.serif"] = "computer modern roman"
rcParams["font.size"] = 12.0

rcParams["axes.linewidth"] = 1

# X-axis
rcParams["xtick.direction"] = "in"
rcParams["xtick.minor.visible"] = True
# Y-axis
rcParams["ytick.direction"] = "in"
rcParams["ytick.minor.visible"] = True

COMPUTE = {
    "flops": 10726000000000.0,
    "compute_bandwidth": 7530482700,
    "memory": 320000000000,
}
DURATION = 18000  # seconds

import argparse

if __name__ == "__main__":

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1001)
    RESULT_FILE = "/home/rwb/Dropbox/University/PhD/experiment_data/chapter3/maximal/all/workflows/8988841495393588671_2024-11-10_21-38-50.csv"
    df = pd.read_csv(RESULT_FILE)

    timestep_array = np.array([1, 5, 15, 30, 60, 60*5])

    timestep_df = pd.DataFrame()

    # Calculate absolute timestep durations 
    for timestep in timestep_array:
        comp_cost = df["fraction_compute_cost"] * 18000 * SI.peta
        series = comp_cost / (COMPUTE["flops"] * timestep)
        timestep_df[timestep] = series.astype(int) * timestep
        timestep_df[timestep] = timestep_df[timestep].replace(0,1).ffill()

    

