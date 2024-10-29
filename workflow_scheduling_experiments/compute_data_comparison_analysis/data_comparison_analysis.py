import sys
import numpy as np
import pandas
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

from skaworkflows.common import SI
# This will store the parameters we used to generate the workflow files
from workflow_scheduling_experiments.basic_experiment.create_observation_plans import LOW_OBSERVATIONS, MID_OBSERVATIONS

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

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# Temporary globals
workflows = ['ICAL', 'DPrepA', 'DPrepB', 'DPrepC', 'DPrepD']

# HPSO and their standard durations (from parameteric model)
# low_hpsos = {'hpso01':18000, 'hpso02a':18000, 'hpso02b':18000}
# mid_hpsos = {'hpso13': 28800 ,'hpso15':15840 , 'hpso22':28800, 'hpso32':7920}

# Setup multipler for a given observation
compute_unit = 10 ** 15  # Peta flop
data_unit = 10 ** 6  # per million visibilites
bytes_per_vis = 12


# Runtime of the parametric model on provisional SDP infrastructure. Taken from parametric model outputs in SKA Workflows code
# par_dict = {'hpso32':706,
# 'hpso22': 62847,
# 'hpso02b':26732,
# 'hpso02a':26732,
# 'hpso13':5655,
# 'hpso01':32090,
# 'hpso15':504}
#

def load_csv(path):
    with path.open() as fp:
        cfg_json = json.load(fp)

    wf_files = []
    for p, wf in cfg_json['instrument']['telescope']['pipelines'].items():
        wf_files.append(wf['workflow'])
    df = pd.read_csv(f"low_maximal/prototype/{wf_files[0]}.csv")


def load_machine_spec_from_config(path: Path) -> list[dict]:
    """
    Open the simulation config path and return all different computing specs

    Parameters
    ----------
    path: path to the config

    Returns
    -------
    machine_specs: list, unique machine specs used in the scheduling
    """
    system_config = {}
    with path.open() as fp:
        system_config = json.load(fp)

    resources = system_config['cluster']['system']['resources']

    str_resources = set([json.dumps(v) for v in resources.values()])

    return [json.loads(s) for s in str_resources]

def load_workflows_from_csvs(path: Path) -> pd.DataFrame:
    df = pd.DataFrame()

    return df

def calculate_relative_compute():
    pass

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(Path(__file__).name, )
    parser.add_argument('path', help="Path to the simulation config file")

    args = parser.parse_args()
    RESULT_PATH = Path(args.path)

    machine_specs = load_machine_spec_from_config(RESULT_PATH)
    flops, compute_bandwidth, memory = machine_specs[-1].values()
    # TODO get workflows from the path in the system config
    workflow = Path()
    all_workflows = load_workflows_from_csvs(workflow)

