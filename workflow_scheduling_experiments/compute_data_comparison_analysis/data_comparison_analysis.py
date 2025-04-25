import argparse
import os
import json
import logging
from pathlib import Path

import numpy as np
# import seaborn as sns
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm, TwoSlopeNorm
from matplotlib.gridspec import GridSpec

from skaworkflows.common import SI

# This will store the parameters we used to generate the workflow files
from workflow_scheduling_experiments.basic_experiment.create_observation_plans import (
    LOW_OBSERVATIONS,
    MID_OBSERVATIONS,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

import matplotlib
matplotlib.use("TkAgg")
# Setup all the visualisation nicities
# rcParams["text.usetex"] = True
rcParams["font.family"] = "serif"
# rcParams["font.serif"] = "computer modern roman"
rcParams["font.size"] = 12.0

rcParams["axes.linewidth"] = 1

# X-axis
rcParams["xtick.direction"] = "in"
rcParams["xtick.minor.visible"] = True
# Y-axis
rcParams["ytick.direction"] = "in"
rcParams["ytick.minor.visible"] = True

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# Temporary globals
pipeline_names = ["ICAL", "DPrepA", "DPrepB", "DPrepC", "DPrepD"]

# HPSO and their standard durations (from parameteric model)
# low_hpsos = {'hpso01':18000, 'hpso02a':18000, 'hpso02b':18000}
# mid_hpsos = {'hpso13': 28800 ,'hpso15':15840 , 'hpso22':28800, 'hpso32':7920}

# Setup multipler for a given observation
compute_unit = 10**15  # Peta flop
data_unit = 10**6  # per million visibilites
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
    for p, wf in cfg_json["instrument"]["telescope"]["pipelines"].items():
        wf_files.append(wf["workflow"])
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

    resources = system_config["cluster"]["system"]["resources"]

    str_resources = set([json.dumps(v) for v in resources.values()])

    return [json.loads(s) for s in str_resources]


def extract_parameters_from_json():
    pass


def load_workflows_from_csvs(config_dir: Path) -> dict:
    """
    DUPLICATED FROM RUN_COMPARISONS_METADATA - MUST CONSOLIDATE
    The difference here is that we want to open the configuration files to extract all
    the workflow file names so we can reference the .csv files. Which is the actual
    information we want


    Also, for the purpose of workflow analysis, we don't care about the different
    timesteps - that's only relevant for the scheduling analysis.
    """

    params = []
    shadow_config = {}
    total_config = 0
    for cfg_path in os.listdir(config_dir):
        if (config_dir / cfg_path).is_dir():
            continue
        total_config += 1
        # Setup for SHADOW config
        timesteps = [1]  # , 5, 15, 30, 60]
        for t in timesteps:
            # shadow_config = config_to_shadow(config_dir / cfg_path)
            # for machine,compute in shadow_config["system"]["resources"].items():
            #     compute['flops'] = compute['flops'] * t
            #     compute['compute_bandwidth'] = compute['compute_bandwidth'] * t
            # shadow_config["system"]["system_bandwidth"]  = shadow_config["system"]["system_bandwidth"]  * t
            # Retrieve workflow parameters

            # TODO consider adding this to SKAWorkflows library as a utility
            with open(config_dir / cfg_path) as fp:
                LOGGER.debug("Path: %s", config_dir / cfg_path)
                cfg = json.load(fp)
            telescope_type = cfg["instrument"]["telescope"]["observatory"]
            pipelines = cfg["instrument"]["telescope"]["pipelines"]
            nodes = len(cfg["cluster"]["system"]["resources"])
            observations = pipelines.keys()
            LOGGER.debug("Observations: %s", observations)
            parameters = (
                pd.DataFrame.from_dict(pipelines, orient="index")
                .reset_index()
                .rename(columns={"index": "observation"})
            )
            parameters["nodes"] = nodes
            parameters["dir"] = config_dir
            # Append information necessary for paramteric runner
            parameters["telescope"] = telescope_type + "-adjusted"
            parameters["timestep"] = t
            for i in range(len(observations)):
                observation = dict(parameters.iloc[i])
                # Observations stored in TOpSim format have _N appended to the end.
                # We do not need this for the scheduling tests
                observation["observation"] = observation["observation"].split("_")[
                    0]
                wf_path = config_dir / observation["workflow"]
                observation["workflow_path"] = wf_path
                with wf_path.open("r") as fp:
                    wf_dict = json.load(fp)
                    workflows = wf_dict["header"]["parameters"]["workflows"]
                    observation["graph_type"] = workflows
                params.append(observation)

    workflow_statistics = pd.DataFrame()
    for observation in params:
        hpso = observation["observation"]
        workflow_data_path = Path(f"{str(observation['workflow_path'])}.csv")
        relevant_keys = ["observation", "duration", "channels", "demand"]
        wf_df = pd.read_csv(workflow_data_path)
        for key in relevant_keys:
            wf_df[key] = observation[key]
        workflow_statistics = pd.concat(
            [workflow_statistics, wf_df], ignore_index=True)

    return workflow_statistics


def calc_compute_time(df: pd.DataFrame):
    """
    Calculate the time it takes to complete the computing requirements of an algorithm

    Notes
    -----
    All < 1 second compute time is round up to 1 second; compute times less than 1 second
    are below the units that we are measuring, and we are only ever going to use discrete
    time intervals when doing simulations.

    Parameters
    ----------
    df: pd.DataFrame, observation dataframe with complete workflow statistics

    Returns
    -------
    Series of costs for the entire dataframe
    """

    return np.ceil(df["fraction_compute_cost"] * df["duration"] * compute_unit / flops)


def calc_data_time(df: pd.DataFrame):
    """
    Calculate the time it takes to read/write the data of an algorithm


    Parameters
    ----------
    df: pd.DataFrame, observation dataframe with complete workflow statistics

    Returns
    -------
    Series of costs for the entire dataframe
    """

    return np.ceil(df["fraction_data_cost"] * df["duration"] * data_unit / compute_bandwidth)


def retrieve_workflow_stats(wf_params: dict):
    """
    For a workflow, get the stats

    Returns
    -------

    """


def calculate_relative_compute():
    pass


def create_computation_dataframe(df):
    """
    Generate a dataframe with compute costs


    Parameters
    ----------
    df

    Notes
    -----
    Excludes computation with 0.0 results from the final dataframe

    All < 1 second compute time is round up to 1 second; compute times less than 1 second
    are below the units that we are measuring, and we are only ever going to use discrete
    time intervals when doing simulations.


    Returns
    -------

    """

    new_df = df.copy()
    compute_time = calc_compute_time(new_df)
    compute_time = compute_time[compute_time != 0.00]
    new_df["Time (s)"] = compute_time.clip(1)
    return new_df


def create_data_dataframe(df):
    """
    Generate a dataframe with compute costs

    Parameters
    ----------
    df

    Returns
    -------

    """
    new_df = df.copy()
    data_time = calc_data_time(new_df)
    data_time = data_time[data_time !=0.00]
    new_df["Time (s)"] = data_time.clip(1)
    return new_df


def calculate_comp_to_data_ratio(df_comp, df_data):
    """
    Determine the computing to data time cost ratio

    > 1 means that the row is data intensive
    < 1 means that the row is computationally intensive

    Parameters
    ----------
    df_comp
    df_data

    Returns
    -------

    """
    df_comp["Ratio"] = np.array(df_data["Time (s)"]) / np.array(df_comp["Time (s)"])
    df_data["Ratio"] = df_comp["Ratio"]
    df_comp.dropna(inplace=True)
    df_data.dropna(inplace=True)
    return df_comp, df_data


def save_processed_workflow_data(workflow_data: pd.DataFrame, source_dir: str):
    """
    Save the processed data frame as a .csv file.

    To reduce the potential of accidentally re-rprocessing, we use the source_dir of the
    data as the root of a hash that forms the file name "processed_<hash>".

    Parameters
    ----------
    workflow_data

    Returns
    -------

    """


def plot_product_cost_variation(df: pd.DataFrame):
    fig = plt.figure()
    gs = GridSpec(1,2, width_ratios=[0.1,0.85])
    # ax.spines['left'].set_position(('data', 1))
    comp_df = create_computation_dataframe(df)
    data_df = create_data_dataframe(df)
    comp_df, data_df = calculate_comp_to_data_ratio(comp_df, data_df)

    telescope = {512:{}, 197: {}}
    # TODO Group by Telescope!

    for group, sub_df in comp_df.groupby(["demand", "workflow_type","product"]):
        demand, workflow_type, product = group
        xaxis_data = sub_df["Ratio"].to_numpy()
        yaxis_data = product
        if workflow_type in telescope[demand]:
            telescope[demand][workflow_type]['x'].append(xaxis_data)
            telescope[demand][workflow_type]['y'].append(yaxis_data)
        else:
            telescope[demand][workflow_type]={'x': [xaxis_data], 'y': [yaxis_data]}


    ax = fig.add_subplot(gs[1])
    # Setup colors using LogNorm
    from matplotlib.colors import SymLogNorm, LogNorm, CenteredNorm

    y = []
    x = np.array([])
    bplot_arrays = []
    for i, y_elem in enumerate(yaxis_data):
        bplot_arrays.append(xaxis_data[i])
        x = np.append(x,xaxis_data[i])
        y.extend([y_elem] * len(xaxis_data[i]))

    # res = ax.boxplot(bplot_arrays,tick_labels=yaxis_data, vert=False,patch_artist=True, boxprops={"facecolor": "bisque"})
    res = ax.scatter(x, y, c=x, cmap="coolwarm",
                     edgecolors="black")
    ax.set_xscale("log")
    # fig.tight_layout()
    # fig.colorbar(res, ax=ax)
    plt.show()



if __name__ == "__main__":
    """
    This script generates the compute-vs-data cost analysis performed on the workflow task 
    and edge costs. The logic is as follows: 

    1. Load the simulation config to derive the available compute used. 
    2. Use this to get workflow spec files for observation in the simulation. 
    """
    parser = argparse.ArgumentParser(
        Path(__file__).name,
    )
    parser.add_argument("path", help="Path to the simulation config file")

    args = parser.parse_args()
    RESULT_PATH = Path(args.path)

    LOGGER.info("Loading machine config...")
    machine_specs = load_machine_spec_from_config(RESULT_PATH)
    flops, compute_bandwidth, memory = machine_specs[-1].values()
    # TODO get workflows from the path in the system config
    # workflow = Path()
    LOGGER.info("Loading workflows...")
    all_workflows = load_workflows_from_csvs(RESULT_PATH.parent)

    # First plot should show distribution of different algorithms (products) as data-intensive or not
    # Look at spines and consider plotting across the +1 x value.
    # Ref: https://jdhao.github.io/2018/05/21/matplotlib-change-axis-intersection-point/
    # Can colour data too? Red is data-intensive, Blue is compute intensive?
    # Use diverging colourscheme

    plot_product_cost_variation(all_workflows)
