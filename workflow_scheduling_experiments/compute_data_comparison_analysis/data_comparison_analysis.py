import os
import json
import logging
from pathlib import Path

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt

from skaworkflows.common import SI
# This will store the parameters we used to generate the workflow files
from workflow_scheduling_experiments.basic_experiment.create_observation_plans import LOW_OBSERVATIONS, MID_OBSERVATIONS

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# Setup all the visualisation nicities
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
pipeline_names = ['ICAL', 'DPrepA', 'DPrepB', 'DPrepC', 'DPrepD']

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

class NpEncoder(json.JSONEncoder):
    # My complete laziness
    # https://java2blog.com/object-of-type-int64-is-not-json-serializable/

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.int64):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, Path):
            return o.name
        return super(NpEncoder, self).default(o)


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

def load_workflows_from_csvs(config_dir: Path) -> pd.DataFrame:
    """
    DUPLICATED FROM RUN_COMPARISONS_METADATA - MUST CONSOLIDATE
    """
    df = pd.DataFrame()

    params = []
    shadow_config = {}
    total_config = 0
    for cfg_path in os.listdir(config_dir):
        if (config_dir / cfg_path).is_dir():
            continue
        total_config += 1
        # Setup for SHADOW config
        timesteps = [1,5,15,30,60]
        for t in timesteps: 
            # shadow_config = config_to_shadow(config_dir / cfg_path)
            # for machine,compute in shadow_config["system"]["resources"].items():
            #     compute['flops'] = compute['flops'] * t
            #     compute['compute_bandwidth'] = compute['compute_bandwidth'] * t
            # shadow_config["system"]["system_bandwidth"]  = shadow_config["system"]["system_bandwidth"]  * t
            # Retrieve workflow parameters

            # TODO consider adding this to SKAWorkflows library

            with open(config_dir / cfg_path) as fp:
                LOGGER.debug("Path: %s", config_dir / cfg_path)
                cfg = json.load(fp)
            telescope_type = cfg["instrument"]["telescope"]["observatory"]
            pipelines = cfg["instrument"]["telescope"]["pipelines"]
            nodes = len(cfg["cluster"]["system"]["resources"])
            observations = pipelines.keys()
            LOGGER.debug('Observations: %s', observations)
            parameters = (
                pd.DataFrame.from_dict(pipelines, orient="index")
                .reset_index()
                .rename(columns={"index": "observation"})
            )
            parameters["nodes"] = nodes
            parameters["dir"] = config_dir
            # Append information necessary for paramteric runner
            parameters["telescope"] = telescope_type + '-adjusted'
            parameters["timestep"] = t
            for i in range(len(observations)):
                observation = dict(parameters.iloc[i])
                # Observations stored in TOpSim format have _N appended to the end.
                # We do not need this for the scheduling tests
                observation['observation'] = observation['observation'].split('_')[0]
                wf_path = config_dir / observation['workflow']
                with wf_path.open('r') as fp: 
                    wf_dict = json.load(fp)
                    workflows = wf_dict['header']['parameters']['workflows']
                    observation['graph_type'] = workflows
                params.append(observation)
            # for o in params:
            #     o["cfg"] = shadow_config

    return params

def retrieve_workflow_stats(wf_params: dict):
    """
    For a workflow, get the stats

    Returns
    -------

    """


def calculate_relative_compute():
    pass


import argparse

if __name__ == "__main__":
    """
    This script generates the compute-vs-data cost analysis performed on the workflow task 
    and edge costs. The logic is as follows: 
    
    1. Load the simulation config to derive the available compute used. 
    2. Use this to get workflow spec files for observation in the simulation. 
    """
    parser = argparse.ArgumentParser(Path(__file__).name, )
    parser.add_argument('path', help="Path to the simulation config file")

    args = parser.parse_args()
    RESULT_PATH = Path(args.path)

    LOGGER.info("Loading machine config...")
    machine_specs = load_machine_spec_from_config(RESULT_PATH)
    flops, compute_bandwidth, memory = machine_specs[-1].values()
    # TODO get workflows from the path in the system config
    # workflow = Path()
    LOGGER.info("Loading workflows...")
    all_workflows = load_workflows_from_csvs(RESULT_PATH.parent)

    for wf in all_workflows:
        print(json.dumps(wf, indent=2, cls=NpEncoder))
