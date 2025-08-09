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
from matplotlib.gridspec import GridSpec

from skaworkflows.common import Telescope

SKA_LOW = Telescope('low')
SKA_MID = Telescope('mid')
# This will store the parameters we used to generate the workflow files
# from workflow_scheduling_experiments.basic_experiment.create_observation_plans import (
#     LOW_OBSERVATIONS,
#     MID_OBSERVATIONS,
# )



LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

import matplotlib

matplotlib.use("TkAgg")
# Setup all the visualisation nicities
# rcParams["text.usetex"] = True
rcParams["font.family"] = "serif"
# rcParams["font.serif"] = "computer modern roman"
rcParams["font.size"] = 6.0

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


def load_system_bandwidth(path: Path):
    system_config = {}
    with path.open() as fp:
        system_config = json.load(fp)
    return system_config["cluster"]["system"]["system_bandwidth"]


def extract_parameters_from_json():
    pass


def load_workflows_from_csvs(config_dir: Path) -> pd.DataFrame:
    """
    DUPLICATED FROM RUN_COMPARISONS_METADATA - MUST CONSOLIDATE
    The difference here is that we want to open the configuration
    files to extract all the workflow file names so we can reference
    the .csv files. Which is the actual information we want


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
                observation["observation"] = observation["observation"].split("_")[0]
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
        workflow_statistics = pd.concat([workflow_statistics, wf_df], ignore_index=True)

    return workflow_statistics


def calc_compute_time(df: pd.DataFrame, telescope_specs: dict):
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
    :param telescope_specs:
    """

    def adjust_value(row):
        if SKA_LOW.is_hpso_for_telescope(row['observation']):
            return row["fraction_compute_cost"] * row["duration"] * compute_unit / telescope_specs['low']['flops']
        else:
            return row["fraction_compute_cost"] * row["duration"] * compute_unit / telescope_specs['mid']['flops']

    return np.ceil(df.apply(adjust_value, axis=1))

    # return np.ceil(df["fraction_compute_cost"] * df["duration"] * compute_unit / flops)


def calc_data_time(df: pd.DataFrame, telescope_specs: dict):
    """
    Calculate the time it takes to read/write the data of an algorithm


    Parameters
    ----------
    df: pd.DataFrame, observation dataframe with complete workflow statistics

    Returns
    -------
    Series of costs for the entire dataframe
    """

    def adjust_value(row):
        if SKA_LOW.is_hpso_for_telescope(row['observation']):
            return row["fraction_data_cost"] * row["duration"] * data_unit / telescope_specs['low']['compute_bandwidth']
        else:
            return row["fraction_data_cost"] * row["duration"] * data_unit / telescope_specs['mid']['compute_bandwidth']

    return  np.ceil(df.apply(adjust_value, axis=1))


def calc_transfer_time(df: pd.DataFrame, telescope_specs: dict):
    """
    Transfer time
    :param df:
    :return:
    """
    def adjust_value(row):
        if SKA_LOW.is_hpso_for_telescope(row['observation']):
            return row["fraction_data_cost"] * row["duration"] * data_unit / telescope_specs['low']['transfer_bandwidth']
        else:
            return row["fraction_data_cost"] * row["duration"] * data_unit / telescope_specs['mid']['transfer_bandwidth']

    return  np.ceil(df.apply(adjust_value, axis=1))


def retrieve_workflow_stats(wf_params: dict):
    """
    For a workflow, get the stats

    Returns
    -------

    """


def calculate_relative_compute():
    pass


def create_computation_dataframe(df, telescope_specs: dict):
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
    compute_time = calc_compute_time(new_df, telescope_specs)
    compute_time = compute_time[compute_time != 0.00]
    print(compute_time)
    new_df["Time (s)"] = compute_time.clip(1)
    return new_df


def create_data_dataframe(df, telescope_specs: dict):
    """
    Generate a dataframe with compute costs

    Parameters
    ----------
    df

    Returns
    -------

    """
    new_df = df.copy()
    data_time = calc_data_time(new_df, telescope_specs)
    data_time = data_time[data_time != 0.00]
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


def calculate_total_cost_with_data(
    df: pd.DataFrame, telescope_specs: dict, twocolumn=True
):
    """
    Using the parametrics method, but using either data-intensive or the compute intensive cost values.

    The expectation is this will give us a large value, and we can cross-check this with the estimates we get for the workflow scheduling.
    :param df:
    :param twocolumn:
    :return:
    """
    if twocolumn:
        fig = plt.figure(
            figsize=(6, 4),
            dpi=300,
        )
    else:
        fig = plt.figure(
            figsize=(10 / 3, 3),
            dpi=300,
        )

    tel_low = Telescope('low')
    tel_mid = Telescope('mid')
    # ax.spines['left'].set_position(('data', 1))
    comp_df = create_computation_dataframe(df, telescope_specs)
    data_df = create_data_dataframe(df, telescope_specs)
    comp_df, data_df = calculate_comp_to_data_ratio(comp_df, data_df)

    comp_df = comp_df.drop_duplicates()
    data_df = data_df.drop_duplicates()
    comp_df["Final Time"] = (
        np.where(
            comp_df["Time (s)"] > data_df["Time (s)"],
            comp_df["Time (s)"],
            data_df["Time (s)"],
        )
        * comp_df["num_tasks"] # At some point I chose fractional for some reason...
    )
    # We don't really need to produce parametric estimates when we have the function. Ah well.
    df_par = pd.read_csv(
        "workflow_scheduling_experiments/basic_experiment/results_2025-07-26.csv",
        index_col=False,
    )
    df_par = df_par.drop_duplicates()

    table_data = []
    for hpso, hpso_df in comp_df.groupby("observation"):
        if tel_low.is_hpso_for_telescope(str(hpso)):
            compute_nodes_used = tel_low.max_compute_nodes
        else:
            compute_nodes_used = tel_mid.max_compute_nodes
        par = df_par[df_par["observation"] == hpso].iloc[0]
        cost = round(sum(hpso_df["Final Time"] / compute_nodes_used))
        table_data.append({'HPSO': hpso, 'Parametric estimate': par['time'], 'Data-included estimate': cost})

    output_df = pd.DataFrame(table_data)
    output_df.to_csv("par_model_data_comparison_basic.csv", float_format="%.2f")

def plot_product_cost_variation(df: pd.DataFrame, telescope_specs: dict, twocolumn=True):
    if twocolumn:
        fig = plt.figure(
            figsize=(6, 4),
            dpi=300,
        )
    else:
        fig = plt.figure(
            figsize=(10 / 3, 3),
            dpi=300,
        )

    # ax.spines['left'].set_position(('data', 1))
    comp_df = create_computation_dataframe(df, telescope_specs)
    data_df = create_data_dataframe(df, telescope_specs)
    comp_df, data_df = calculate_comp_to_data_ratio(comp_df, data_df)

    comp_df = comp_df.drop_duplicates()

    # TODO Group by Telescope!
    import matplotlib.colors as mcolors

    css_colors = list(mcolors.CSS4_COLORS.keys())
    workflows = ["DPrepA", "DPrepB", "DPrepC", "DPrepD", "ICAL"]

    _legend = workflows
    incr = len(mcolors.CSS4_COLORS) % len(workflows)
    legend_colors = {}
    for i in range(len(_legend)):
        legend_colors[_legend[i]] = css_colors[i * incr]

    low_df = comp_df[comp_df["demand"] == 512]
    mid_df = comp_df[comp_df["demand"] == 197]

    count = 0

    gs = GridSpec(
        2, 4, right=0.875, hspace=0.4, wspace=0.2
    )  # , hspace=0.5, top=0.9, wspace=0, right=0.85)  # , width_ratios=[0.1,0.85]
    curr_handles = {}
    for hpso in comp_df.groupby("observation"):
        hpso, group_df = hpso
        ax = fig.add_subplot(gs[count])

        results = {}
        for group, sub_df in group_df.groupby(["workflow_type", "product"]):
            workflow_type, product = group
            xaxis_data = sub_df["Ratio"].to_numpy()
            split_product = product.split(" ")
            if len(split_product) > 1:
                product = split_product[0] + "*"
            if "LSM" in product:
                product = "Update*"
            if workflow_type in results:
                if product in results[workflow_type]:
                    results[workflow_type][product].append(xaxis_data)
                else:
                    results[workflow_type][product] = xaxis_data
            else:
                results[workflow_type] = {product: xaxis_data}

        for wf, xy in results.items():
            # Sort the keys (y-values)
            sorted_items = sorted(xy.items())  # Sorted by y-value alphabetically

            # Unpack into sorted y and x values
            y = [k for k, v in sorted_items]
            x = [v[0] for k, v in sorted_items]
            # for i, _y in enumerate(y_sorted):
            #     x = x_sorted[i]
            #     y = [_y]*len(x)
            x.reverse()
            y.reverse()
            ax.scatter(
                x,
                y,
                label=wf,
                color=legend_colors[wf],
                edgecolors="black",
                linewidth=1,
                s=15,
            )

        if count % 4 > 0:
            ax.get_yaxis().set_visible(False)

        ax.set_xscale("log")
        ax.vlines(1, 0, 13, linestyle="dashed", color="grey", zorder=-1)
        ax.set_xbound(1e-3, 100)
        ax.set_title(f"{hpso.upper()}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Algorithm")
        h, l = ax.get_legend_handles_labels()
        by_label = dict(zip(l, h))
        if len(by_label) > len(curr_handles):
            curr_handles = by_label

        count += 1
        if count == 3:
            count+=1
        # ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    plt.legend(
        curr_handles.values(),
        curr_handles.keys(),
        title='Workflow',
        # fontsize="small",
        bbox_to_anchor=(0.8, 2.0),
    )
    # handles, labels = plt.gca().get_legend_handles_labels()

    # fig.tight_layout()
    # fig.colorbar(res, ax=ax)

    plt.savefig("product_cost_var.png")


def plot_supporting_data_variation(df: pd.DataFrame, telescope_specs, twocolumn=False):
    if twocolumn:
        fig = plt.figure(
            figsize=(6, 4),
            dpi=300,
        )
    else:
        fig = plt.figure(
            figsize=(10 / 3, 3),
            dpi=300,
        )

    # ax.spines['left'].set_position(('data', 1))
    comp_df = create_computation_dataframe(df, telescope_specs)
    data_df = create_data_dataframe(df, telescope_specs)
    comp_df, data_df = calculate_comp_to_data_ratio(comp_df, data_df)
    comp_df = comp_df.drop_duplicates()

    comp_df["transfer_data_time"] = calc_transfer_time(comp_df, telescope_specs)

    df_comp = comp_df.sort_values(by="observation")
    df_comp_dataintensive = df_comp[df_comp["Ratio"] >= 1]
    df_comp_dataonly = df_comp_dataintensive[
        df_comp_dataintensive["transfer_data_time"] >= 1
    ]

    ax = fig.add_subplot()
    y = []
    x = []
    hpsos = set()
    x_jitter = []
    hpsos = sorted(df_comp_dataonly["observation"].unique())
    category_to_x = {hpso: i for i, hpso in enumerate(hpsos)}
    for hpso, group in df_comp_dataonly.groupby("observation"):
        # x.append(hpso.upper()) # uncomment if you want box
        xpos = category_to_x[hpso]
        jitter = np.random.normal(loc=0, scale=0.1, size=len(group))
        x_jitter.extend(xpos + jitter)
        y.extend(group["transfer_data_time"])
        # y.append(group['transfer_data_time'])
    # bplot = ax.boxplot(y, patch_artist=True, tick_labels=x, whis=(0, 100))
    ax.scatter(x_jitter, y, color="lightgrey", edgecolor="black", s=15)
    # for patch, color in zip(bplot['boxes'], colors*len(bplot['boxes'])):
    #     patch.set_facecolor(color)

    # for pc in parts['bodies']:
    #     pc.set_facecolor('grey')
    #     pc.set_edgecolor('black')
    #     pc.set_alpha(1)

    ax.set_xlabel("Observation type")
    ax.set_ylabel("Time (s)")
    ax.set_yscale("log")
    plt.xticks(ticks=range(len(hpsos)), labels=[hp.strip("hpso") for hp in hpsos])
    plt.savefig("supporting_data_transfer.png")
    # plt.show()

def compare_methods():
    df = pd.read_csv(
            "workflow_scheduling_experiments/basic_experiment/results_2025-07-18.csv",
            index_col=False,
        )
    df_par = pd.read_csv(
        "workflow_scheduling_experiments/basic_experiment/results_2025-07-31.csv",
        index_col=False,
    )
    pd.concat([df, df_par])

    df = df.drop_duplicates()
    # heft = df[df['method'] == 'heft']
    # fcfs = df[df['method'] == 'fcfs']
    pivoted = df.pivot_table(
        index=['observation', 'data', 'data_distribution'],
        columns='method',
        values='time'
    ).reset_index()
    # speedup = ((fcfs['time']-heft['time'])/fcfs['time'])*100
    # TODO consider modifying data/data_distribution to be more reader-friendly
    pivoted['speedup'] = ((pivoted['fcfs']-pivoted['heft']) / pivoted['fcfs'])*100
    pivoted['data'] = pivoted['data'].replace(False, "")
    pivoted['data'] = pivoted['data'].replace(True, "\\checkmark")
    pivoted['data_distribution'] = pivoted['data_distribution'].replace("standard", "")
    pivoted['data_distribution'] = pivoted['data_distribution'].replace("edges", "\\checkmark")
    # pivoted = pivoted.sort_values(by='data_distribution', ascending=False)
    pivoted.to_csv("heft_fcfs_comparison.csv", float_format="%.2f")

def plot_scheduling_comparisons(method:str):
    df = pd.read_csv(
        "workflow_scheduling_experiments/basic_experiment/results_2025-07-18.csv",
        index_col=False,
    )
    df = df.drop_duplicates()
    df = df[df['method']==method]
    alt_df = df[df['method']!=method]
    df_par = pd.read_csv(
        "workflow_scheduling_experiments/basic_experiment/results_2025-07-26.csv",
        index_col=False,
    )
    df_par = df_par.drop_duplicates()
    fig = plt.figure(
        figsize=(3, 2),
        dpi=300,
    )
    ax = fig.subplots()
    plt.subplots_adjust(bottom=0.15)
    # ax2 = fig.subplots()
    i = 0
    df = df.sort_values(by=["data_distribution"], ascending=False)
    # nfig = plt.figure(figsize=(10/3, 3), dpi=300)
    # alt_ax = nfig.subplots()
    width = 0.20
    difference_dict = {}
    import matplotlib.colors as mcolors

    css_colors = list(mcolors.CSS4_COLORS.keys())
    # NEED TO SPLIT BY HPSO

    incr = len(mcolors.CSS4_COLORS) % 5
    colors = css_colors[10::4]

    # for group, group_df in df.groupby(['data_distribution', 'data']):
    groups = []
    groups.append(
        (
            ("standard", False),
            df[(df["data_distribution"] == "standard") & (df["data"] == False)],
        )
    )
    groups.append(
        (
            ("standard", True),
            df[(df["data_distribution"] == "standard") & (df["data"] == True)],
        )
    )
    groups.append(
        (
            ("edges", True),
            df[(df["data_distribution"] == "edges") & (df["data"] == True)],
        )
    )
    for g in groups:
        type, group_df = g
        distribution, data = type
        obs = []
        y = []
        par = []
        for observation, ngroup_df in group_df.groupby("observation"):
            ngroup_df.sort_values(by=["timestep"])
            base_time = ngroup_df[group_df["timestep"] == 1]["time"].iloc[0]
            timestep = ngroup_df["timestep"]
            time = ngroup_df["time"] * timestep
            difference = (time - base_time) / base_time
            # if distribution == 'standard' and data:
            #     alt_ax.scatter(x=timestep, y=difference, label=observation)
            #     alt_ax.legend()
            # ax.scatter(x=[observation], y=base_time, label=observation, marker='o')
            par_time = df_par[
                (df_par["data"] == data)
                & (df_par["data_distribution"] == distribution)
                & (df_par["observation"] == observation)
            ]["time"].iloc[0]
            par.append(1.0)
            obs.append(observation.strip('hpso'))
            y.append(base_time / par_time)
            if f"{data}_{distribution}" in difference_dict:
                difference_dict[f"{data}_{distribution}"].append(base_time / par_time)
            else:
                difference_dict[f"{data}_{distribution}"] = [base_time / par_time]
            # ax.scatter(x=[observation], y=par_time, label=observation, marker='v')
            # ax.set_title(f"{data} + {distribution}")
        x = np.arange(len(obs))
        if i == 0:
            offset = width * i
            ax.bar(
                x=x + offset,
                height=par,
                width=width,
                label=f"Par Model",
                facecolor=colors[i],
                edgecolor="black",
            )
            i += 1
        offset = width * i
        ax.bar(
            x=x + offset,
            height=y,
            width=width,
            label=f"{data} + {distribution}",
            facecolor=colors[i],
            edgecolor="black",
        )
        ax.set_xticks(x + width, obs)
        ax.set_ylabel("Final Schedule (s): Parametric Estimate (s)")
        ax.set_xlabel("HPSO")
        ax.legend()

        i += 1
    ax.set_ylim(0, 8)
    plt.savefig(f"{method}_scheduling_results.png")
    # plt.show()

    difference_dict["obs"] = obs
    pd.DataFrame(difference_dict).to_csv(f"Difference_Schedule_{method}.csv", float_format="%.2f")


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
    parser.add_argument("--low", help="Path to the low simulation config file")
    parser.add_argument("--mid", help="Path to the mid simulation config file")

    args = parser.parse_args()
    low_path = Path(args.low)
    mid_path = Path(args.mid)

    LOGGER.info("Loading machine config...")
    telescope_specs = {"low": {}, "mid": {}}
    # SKALow
    machine_specs = load_machine_spec_from_config(low_path)
    # flops, bandwidth, memory = machine_specs[-1].values()
    telescope_specs["low"].update(
        {"flops": machine_specs[0]["flops"], "compute_bandwidth": machine_specs[0]["compute_bandwidth"], "memory":
            machine_specs[0]["memory"]}
    )
    telescope_specs["low"]["transfer_bandwidth"] = load_system_bandwidth(low_path)
    # SKAMid
    machine_specs = load_machine_spec_from_config(mid_path)
    # flops, bandwidth, memory = machine_specs[-1].values()
    telescope_specs["mid"].update(
        {"flops": machine_specs[0]["flops"], "compute_bandwidth": machine_specs[0]["compute_bandwidth"], "memory":
            machine_specs[0]["memory"]}
    )
    telescope_specs["mid"]["transfer_bandwidth"] = load_system_bandwidth(mid_path)

    # workflow = Path()
    LOGGER.info("Loading workflows...")
    # Doesn't matter which path we provide here as we are using it to get the directory
    all_workflows = load_workflows_from_csvs(low_path.parent)

    # First plot should show distribution of different algorithms (products) as data-intensive or not
    # Look at spines and consider plotting across the +1 x value.
    # Ref: https://jdhao.github.io/2018/05/21/matplotlib-change-axis-intersection-point/
    # Can colour data too? Red is data-intensive, Blue is compute intensive?
    # Use diverging colourscheme

    # plot_product_cost_variation(all_workflows, telescope_specs)
    # plot_supporting_data_variation(all_workflows, telescope_specs)
    # plot_scheduling_comparisons('fcfs')
    # plot_scheduling_comparisons('heft')
    # calculate_total_cost_with_data(all_workflows, telescope_specs)
    compare_methods()
    # plt.show()
