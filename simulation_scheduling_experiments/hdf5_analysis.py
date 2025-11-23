#!/usr/bin/env python
# coding: utf-8
import datetime
import logging
import random
import shutil
import sys
import numpy as np
# import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import itertools
from pathlib import Path
from io import StringIO
from matplotlib import axes
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from matplotlib import rcParams
from sympy.solvers.diophantine.diophantine import find_DN

from skaworkflows.common import SKALow, Telescope

###########################################################
# PLOTTING PARARMETERS
###########################################################

# Setup all the visualisation nicities
rcParams["text.usetex"] = False
rcParams["font.family"] = "serif"
# rcParams['font.serif'] = "computer modern roman"
rcParams["font.size"] = 9.0

rcParams["axes.linewidth"] = 1

# X-axis
rcParams["xtick.direction"] = "in"
rcParams["xtick.minor.visible"] = True
# Y-axis
rcParams["ytick.direction"] = "in"
rcParams["ytick.minor.visible"] = True

###########################################################
# GLOBALs
###########################################################

logging.basicConfig(level="INFO")
LOGGER = logging.getLogger()

# Constants from updated SDP Cost Estimates (Alexander 2016)
TOTAL_COMPUTE_LOW_HPSOS_FLOPS = 1.2e24
LOW_HPSO_DURATION_YEARS = 2.8
MAXIMAL_OBSERVATION_COMPUTE_FLOPS = 1.5e21
SDP_AVERAGE_COMPUTE_FLOPS = 13.8e15
LOW_SDP_AVERAGE_COMPUTE_FLOPS_UPDATED = 9.623
LOW_REALTIME_RESOURCES = 164
MID_REALTIME_RESOURCES = 281


class TopSimResult:
    DATASET_TYPES = ["sim", "summary", "tasks"]  # TODO change to 'Result Tables'

    def __init__(result_path: str):
        store = pd.HDFStore(result_path)
        result_path: Path = Path(result_path)
        config_path: Path = None
        timestep: int = None
        observation_plan: pd.DataFrame = None


###########################################################
# DATA
###########################################################

def extract_simulations_from_hdf5(result_paths, verbose=True):

    simulations = {}
    for result_path in result_paths:
        if not result_path.exists():
            raise FileNotFoundError(f"{result_path}")

        if result_path.is_dir():
            count = 5
            for p in result_path.iterdir():
                if count < 0:
                    yield simulations
                    simulations = {}
                    count = 10
                else:
                    # TODO if this is a folder iterate through all hdf5 files
                    tmp_simulations = {}
                    store = pd.HDFStore(str(p))
                    keysplit = []
                    for k in store.keys():
                        keysplit.append(k.split("/"))
                    store.close()
                    if verbose:
                        print(p, keysplit)
                    dataset_types = ["sim", "summary", "params"]
                    tmp_simulations.update(
                        {f"{e[1]}/{e[2]}": {d: None for d in dataset_types} for e in keysplit}
                    )
                    for simulation, dtype in tmp_simulations.items():
                        for dst in dataset_types:
                            tmp_simulations[simulation][dst] = pd.read_hdf(
                                p, key=f"{simulation}/{dst}"
                            )
                    simulations.update(tmp_simulations)
                    if verbose:
                        for keys in tmp_simulations.keys():
                            print(keys)
                    count -= 1
            yield simulations


all_pairs = list(itertools.product(SKALow.baselines, SKALow.stations))
SKALOW_LARGE_PAIRS = [(65000, 256), (32500, 512), (65000, 512)]
SKALOW_MED_PAIRS = [(65000, 64), (65000, 128), (32500, 128), (32500, 256), (16250, 512), (16250, 256), (8125, 512)]
SKALOW_SMALL_PAIRS = list(set(all_pairs) - set(SKALOW_MED_PAIRS) - set(SKALOW_LARGE_PAIRS))


def get_observation_plan_size(df):
    """
    Determine if the dataframe demand tuples match the different 'size' types above.

    We use the isdisjoint to determine if there are _any_ examples of the different tuples as
    defined above.

    :param df:
    :return:
    """
    obs = [tuple(row) for index, row in df[['baseline', 'demand']].iterrows()]
    # Calculate the percentage of each small/medium/large there is in the observation set
    counts = {"large": 0, "medium": 0, "small": 0}
    for o in obs:
        if o in SKALOW_LARGE_PAIRS:
            counts["large"] += 1
        elif o in SKALOW_MED_PAIRS:
            counts["medium"] += 1
        else:
            counts["small"] += 1

    total = len(obs)

    for key, value in counts.items():
        counts[key] = round(value / total, 2)

    return counts


def collate_simulation_results(parent_dir: Path, simulations: dict):
    # TODO consider applying timesteps to everything here so we don't have to later
    df_total = pd.DataFrame()
    processed = []
    for simulation, dtype in simulations.items():
        logging.info("Collating simulation results for %s", simulation)
        df = dtype["summary"]
        df_tel = df[(df["actor"] == "instrument")]
        obs_durations = []
        obs_length = 0
        # TODO does this only work for non-concurrent observations?
        for obs in set(df_tel["observation"]):
            df_obs = df_tel[df_tel["observation"] == obs]
            obs_length = (
                    df_obs[df_obs["event"] == "finished"]["time"].iloc[0]
                    - df_obs[df_obs["event"] == "started"]["time"].iloc[0]
            )
            obs_durations.append(obs_length)

        df_sim = dtype["sim"]
        df_params = dtype["params"]

        # Get the simulation parameters from the configuration file.
        cfg_path = Path(df_sim["config"].iloc[0])
        if cfg_path in processed:
            continue
        # cfg_path = cfg_path.parent / str(cfg_path.parent / 'processed' / cfg_path.name)
        cfg_path = parent_dir / cfg_path.name
        with open(cfg_path, "r", encoding="utf-8") as fp:
            cfg = json.load(fp)
        timestep = cfg["timestep"]
        pipelines = cfg["instrument"]["telescope"]["pipelines"]
        resources = cfg['cluster']['system']['resources']
        str_resources = set([json.dumps(v) for v in resources.values()])

        nodes = [json.loads(s) for s in str_resources]

        parameters = (
            pd.DataFrame.from_dict(pipelines, orient="index")
            .reset_index()
            .rename(columns={"index": "observation"})
        )
        observations = pipelines.keys()

        # print(f"{parameters['workflow']}")
        parameters["timestep"] = timestep

        # TODO consider getting the schedule length the same we get the obs_durations, so it's purely reflecting
        # The time each workflow was computing.
        # The only problem here is we will need to ensure we don't double count overlapping workflows.
        parameters["schedule_length"] = calculate_total_computing_time(df, timestep)  # len(df_sim)
        parameters["planning"] = df_sim["planning"]
        parameters["scheduling"] = df_sim["scheduling"]
        parameters["max_running_tasks"] = df_sim[
            "running_tasks"].max()  # Can multiply each entry by 5 to get the time step to report on
        parameters["min_running_tasks"] = df_sim["running_tasks"].min()
        parameters["mean_running_tasks"] = df_sim["running_tasks"].sum() / parameters['schedule_length'].iloc[0]
        parameters["mean_ingest_demand"] = df_sim["ingest_resources"].sum() / parameters['schedule_length'].iloc[0]
        parameters["max_ingest_demand"] = df_sim["ingest_resources"].max()
        # TODO it's really inefficient having so many tables with the same number. consider a lookup table with these sorts
        # of global parameters per-config (Did I just invent a type of database???)
        parameters["nodes"] = [json.dumps(nodes)] * len(parameters["schedule_length"])
        # parameters["observation_size"] = get_observation_plan_size(parameters)
        # Use simulation config to differentiate between different sims
        parameters["sim_cfg"] = cfg_path.name
        parameters["total_obs_duration"] = sum(obs_durations)
        parameters["simulation_run"] = simulation
        counts = get_observation_plan_size(parameters)
        parameters["large"] = counts['large']
        parameters["medium"] = counts['medium']
        parameters["small"] = counts['small']

        parameters["use_task_data"] = [df_params["use_task_data"].iloc[0]]*len(parameters)
        parameters["use_edge_data"] = [df_params["use_edge_data"].iloc[0]]*len(parameters)

        df_total = pd.concat([df_total, parameters], ignore_index=True)

    return df_total


def calculate_total_computing_time(simulation_df, timestep):
    scheduler_start = []
    scheduler_finish = []
    obs = set(simulation_df["observation"])
    obs_d = {o: {} for o in obs}
    obs_list = [[], [], []]
    for o in obs_d:
        obs_d[o]["scheduler"] = simulation_df[
            (simulation_df["observation"] == o) & (simulation_df["actor"] == "scheduler") & (
                        simulation_df['resource'] == 'allocation')
            ]

    for o in sorted(obs):
        obs_list[0].append(f"{o}")  # Scheduler
        sdf = obs_d[o]["scheduler"]
        scheduler_start.append(
            int(sorted(sdf[sdf["event"] == "started"]["time"])[0])
        )
        scheduler_finish.append(
            int(sorted(sdf[sdf["event"] == "stopped"]["time"])[-1])
        )

    intervals = list(zip(scheduler_start, scheduler_finish))

    # Sanity check
    if not intervals:
        return 0

    # Sort by start time
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]

    for current_start, current_end in intervals[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end:
            # Overlapping → merge
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            # No overlap → add new interval
            merged.append((current_start, current_end))

    # Sum total duration of merged intervals
    return sum([end - start for start, end in merged])


def process_workflow_stats(base_dir: str, df_total: pd.DataFrame):
    """
    Go through each workflow config file related to the directory, and get summary
    information from them to describe the compute requirements for a given config.

    Parameters
    ----------
    config_dir

    Returns
    -------

    """

    # TODO also produce stats for total_data

    workflow_paths = set(df_total["workflow"])
    # total_workflow_df = pd.DataFrame()
    total_compute = 0
    total_duration = 0
    max_compute = 0
    peak_compute = 0
    for index, row in df_total.iterrows():
        duration = row["duration"]
        total_duration += duration
        wf_path = Path(base_dir) / (row["workflow"])
        with wf_path.open() as fp:
            jdict = json.load(fp)
            baseline = jdict["header"]["parameters"]["baseline"]
        csv_path = wf_path.with_suffix(".csv")
        if not csv_path.exists():
            return 0
        total_workflow_df = pd.read_csv(csv_path)
        if "total_compute" in total_workflow_df:
            compute = sum(total_workflow_df["total_compute"]) * duration
            # This would be the compute for a given workflow
        else:
            # Pulsar is entire cost for the whole workflow
            compute = total_workflow_df.iloc[0]['total_cost'] * duration / (10 ** 15)

        total_compute += compute
        peak_compute = max(max_compute, compute)

    return total_compute, peak_compute, baseline


def convert_categorical_ints_to_str(df_total: pd.DataFrame):
    """
    Some of our variables are integers but we want to treat them like categories so there
    are not 'unecessary' spaces in our plots.

    Converting to strings allows us to fix them in a sorted order without numeric spacing.
    Parameters
    ----------
    df: data frame we want to sort

    Returns
    -------
    df,
    """
    df_total = df_total.sort_values(by="demand")
    df_total["demand"] = df_total["demand"].astype("str")
    # df_total['demand'] = sorted(df_total['demand'].astype('str'), key=int)
    return df_total


def pretty_print_simulation_results(simulations, key, verbose=False):
    """
    Get final duration of simulation and whether or not it was successful.

    Produce a 'table' of parameters to help differentiate what was in the HDF5
    Parameters
    ----------
    simulation
    key

    Returns
    -------
    None: Prints output to terminal
    """
    df = simulations[key]["sim"]
    cfg_path = Path(df["config"].iloc[0])
    with open(cfg_path) as fp:
        cfg = json.load(fp)
    pipelines = cfg["instrument"]["telescope"]["pipelines"]
    nodes = len(cfg["cluster"]["system"]["resources"])

    # Determine if plan was successful
    obs_durations = get_observation_duration(simulations[key])
    df_sum = simulations[key]["summary"]
    df_sched = df_sum[(df_sum["actor"] == "scheduler")]

    success = True
    second_last_index = -2
    if len(obs_durations) < 2:
        second_last_index = -1
    if (
            sum(obs_durations)
            - sorted(df_sched[df_sched["event"] == "stopped"]["time"])[second_last_index]
    ) < 0:
        success = False

    parameters = (
        pd.DataFrame.from_dict(pipelines, orient="index")
        .reset_index()
        .rename(columns={"index": "observation"})
    )
    parameters["nodes"] = nodes  # Number of nodes

    parameters["schedule_length"] = len(df)
    parameters["planning"] = df["planning"]
    parameters["scheduling"] = df["scheduling"]
    parameters["success"] = success
    parameters = parameters.drop(columns=["workflow", "workflow_type", "graph_type"])
    if verbose:
        print(parameters)
    return parameters


def create_simulation_schedule_map(simulation_df):
    simulation_df = simulation_df  # Remove this at some point this
    actors = set(simulation_df["actor"])
    # Observation telescope, started/finished
    # observation buffer, start/end -> we don´t particularly care about buffer
    # observation scheduler, added/removed
    obs = set(simulation_df["observation"])
    inst, sched = {}, {}
    obs_d = {o: {} for o in obs}
    for o in obs_d:
        obs_d[o]["telescope"] = simulation_df[
            (simulation_df["observation"] == o) & (simulation_df["actor"] == "instrument") & (
                        simulation_df['resource'] == 'telescope')
            ]
        obs_d[o]["buffer"] = simulation_df[
            (simulation_df["observation"] == o) & (simulation_df["actor"] == "buffer") & (
                    simulation_df['resource'] == 'transfer')
            ]
        obs_d[o]["scheduler"] = simulation_df[
            (simulation_df["observation"] == o) & (simulation_df["actor"] == "scheduler") & (
                        simulation_df['resource'] == 'allocation')
            ]

    simulation_total_time = 0
    # begin, end = [], []
    obs_list = [[], [], []]
    scheduler_start = []
    scheduler_end = []
    buffer_start = []
    buffer_end = []
    telescope_start = []
    telescope_end = []
    for o in sorted(obs):
        obs_list[0].append(f"{o}")  # Scheduler
        sdf = obs_d[o]["scheduler"]
        scheduler_start.append(
            int(sdf[sdf["event"] == "started"]["time"].iloc[0]) * 5 / 3600
        )
        scheduler_end.append(
            int(sdf[sdf["event"] == "stopped"]["time"].iloc[0]) * 5 / 3600
        )

        # Buffer transfer events may not happen
        obs_list[2].append(f"{o}")  # Buffer
        bdf = obs_d[o]["buffer"]
        if bdf.empty:
            buffer_start.append(0)
            buffer_end.append(0)
        else:
            # Loop through the start/stop times
            start = []
            end = []
            for i in range(int(len(bdf) / 2)):
                start.append(bdf[bdf['event'] == 'started']['time'].iloc[i])
                end.append(bdf[bdf['event'] == 'stopped']['time'].iloc[i])
            #
            buffer_start.append(min(start) * 5 / 3600)
            buffer_end.append(max(end) * 5 / 3600)

            # buffer_start.append(
            #     int(bdf[bdf["event"] == "started"]["time"].iloc[0]) * 5 / 3600
            # )
            # buffer_end.append(
            #     int(bdf[bdf["event"] == "stopped"]["time"].iloc[0]) * 5 / 3600
            # )

        obs_list[1].append(f"{o}")  # Telescope
        tdf = obs_d[o]["telescope"]
        telescope_start.append(
            int(tdf[tdf["event"] == "started"]["time"].iloc[0]) * 5 / 3600
        )
        telescope_end.append(
            int(tdf[tdf["event"] == "finished"]["time"].iloc[0]) * 5 / 3600
        )

    # TODO mark pulsars as scatter plot values and overlay on axis. Makes it a lot easier to see
    group_labels = ["Scheduler", "Telescope", "Buffer"]
    # import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.subplots()
    # fig, ax = plt.subplots()
    values = [[scheduler_start, scheduler_end], [telescope_start, telescope_end], [buffer_start, buffer_end]]
    for i, (actor, group_label) in enumerate(zip(values, group_labels)):
        start, end = actor
        rects = ax.barh(
            range(i, len(start) * 3, 3),
            np.array(end) - np.array(start),
            label=group_label,
            left=np.array(start),
        )

        ax.set_yticks(range(i, len(start) * 3, 3), obs_list[i])
        # break
        # ax.barh(range(len(begin)), np.array(end) - np.array(begin),
        #         color=['grey', 'orange'],
        #         left=np.array(begin), edgecolor='black')
    ax.legend()
    # plt.savefig(f"ScheduleMap_{hash(key)}.png")
    return simulation_total_time


def get_observation_duration(df):
    df_tel = df[(df["actor"] == "instrument")]

    for obs in set(df_tel["observation"]):
        df_obs = df_tel[df_tel["observation"] == obs]
        print(
            df_obs[df_obs["event"] == "finished"]["time"].iloc[0]
            - df_obs[df_obs["event"] == "started"]["time"].iloc[0]
        )


def calculate_low_percentages():
    pass


def calculate_mid_percentages():
    pass


def calculate_week_on_telescope_comparison_stats(plan_total_compute: float,
                                                 plan_peak_compute: float):
    """
    The design estimates for the SDP state that a 2.8 year time-on-telescope HPSO program
    will require 1.2e24 FLOPs (total_flops) to process the data (Alexander et al., 2016).
    This is where the 13.6PFLOP minimum average compute comes from  (total_flops / 2.8 years-in-seconds).

    Using the total_flops value, we can estimate the weekly average output expected:

        total_flops / (2.8 * 52) ~= 8.24e+21 (weekly_estimate)

    We can use _this_ value to compare against our own generated observation plans.
    Whatever the total compute required for that plan will be a percentage of the weekly estimate,
    and we can use this to demonstrate where abouts that particular plan is relative to the
    estimated average produced during the system sizing.

    We will do the same for the estimated maximal case of 1.5e21 (maximal_estimate), and our plan_peak_compute
    value, which is the maximal observation within the observing plan.

    Ref: Alexander 2016, Updated SDP Cost Basis of  Estimate June 2016

    Notes
    -----
    The returned dictionary will have the following keys:

    "plan_total_compute": Total compute of the plan, in FLOPS
    "relative_total_compute": plan_total_compute / weekly_estimate
    "plan_peak_compute": Peak observation compute demand for the plan
    "relative_peak_compute": plan_peak_compute / maximal_estimate

    :return: dict
    """
    return {"plan_total_compute": plan_total_compute,
            "relative_total_compute": plan_total_compute / (MAXIMAL_OBSERVATION_COMPUTE_FLOPS / 1e15),
            "plan_peak_compute": plan_peak_compute,
            "relative_peak_compute": plan_peak_compute / (TOTAL_COMPUTE_LOW_HPSOS_FLOPS / 1e15)}


def calculate_average_flops_in_plan(plan_total_compute,
                                    schedule_length_seconds,
                                    average_flops,
                                    average_running_tasks,
                                    max_running_tasks):
    """
    The SDP estimates identified the minium average compute required as 13.6PFLOP/sec.
    They used this in their costing calculations as the lower bounds on the capacity a provisioned SDP
    would have; their final result was an SDP that provided an average of 13.8PFLOP/sec.

    We can take the plan_total_compute and find our own average compute based on the
    time taken to do the computing in our simulation.

    From this, we can compare the plan_average_compute_from_flops to the SDP average compute. This will (hopefully) help us
    explain _why_ the average compute is lower than what we would expect.

    :return:  dictionary of the "plan_average_compute_from_flops" and "plan_relative_average_compute" values.
    """
    plan_average_compute_from_flops = plan_total_compute / (schedule_length_seconds - (18000 * 2))
    plan_average_compute_from_nodes = average_flops * average_running_tasks
    plan_peak_compute_from_nodes = average_flops * max_running_tasks
    return {"plan_average_compute_from_flops": plan_average_compute_from_flops,
            "plan_average_compute_from_nodes": plan_average_compute_from_nodes / 1e15,
            "plan_peak_compute_from_nodes": plan_peak_compute_from_nodes / 1e15,
            "relative_average_compute": plan_average_compute_from_flops / (SDP_AVERAGE_COMPUTE_FLOPS / 1e15)}


def get_compute_node_statisics(nodes):
    """
    There may be more than one type of compute node, which requires us to calculate the average computing available
    across the whole simulation.

    The data in nodes will look something like:

        >>> [{'flops': 10726000000000.0,
        >>>     'compute_bandwidth': 7530482700,
        >>>     'memory': 320000000000}]

    :param nodes: list
    :return: mean_node_flops, mean_node_bandwidth
    """
    flops = [d["flops"] for d in nodes]
    compute_data_bandwidth = [d["compute_bandwidth"] for d in nodes]
    mean_node_flops = np.mean(flops)
    mean_node_bandwidth = np.mean(compute_data_bandwidth)

    return mean_node_flops, mean_node_bandwidth


def calculate_demand_percentage(subset_df: pd.DataFrame, telescope: str = 'low'):
    """
    Apply quadratic relationship to demand bins


    :param demand: array of demand across an observing plan
    :param telescope: string that indicates which telescope
    :return:
    """

    t = Telescope(telescope)
    used_stations = subset_df['duration'] * subset_df['demand']
    used_baseline = subset_df['duration'] * subset_df['baseline']
    potential_stations = sum(subset_df['duration']) * t.max_stations
    potential_baselines = sum(subset_df['duration']) * t.max_baseline

    x1 = sum(used_stations) / potential_stations
    # x2 = sum(used_baseline) / potential_baselines
    # return 1-((1-x1) * (1-x2))

    # demand_bins = np.append(t.stations, 513)
    # demand = 0
    # counts, bins = np.histogram(demand, demand_bins)
    # usage = counts*bins[:-1]**2
    # potential = counts.sum() * t.max_stations
    return x1  # sum(used) / potential


def produce_summary_dataframe(df_total, base_dir: Path, verbose=True):
    """
    Determine the resource usage of the telescope across the entire observation plan,
    as a fraction of the maximum possible value.

    Note
    -------
    This is to facilitate differentiating between various observation plan's use of the telescope.

    Currently this assumes LOW telescope

    The following data is stored in the Data Frame:
        cfg,
        baseline,
        demand_ratio,
        channels_ratio,
        data,
        data_distribution,
        success,
        schedule_length,
        success_ratio,
        planning,
        schedule_length_ratio,
        total_compute,
        average_compute,
        computing_to_observation_length_ratio,
        peak_compute,

    """
    max_channels = 128 * 256
    max_demand = 512
    group = df_total.groupby(["sim_cfg", "planning", "simulation_run"])  # , data"])
    # Isolating observing plans, with planning algorithms
    count = 0
    usage = []
    for name, g in group:
        cfg, planning, sim_run = name
        timestep = g["timestep"].iloc[0]
        LOGGER.info("Processing Simulation %s with method %s", sim_run, planning)
        curr_total = df_total[(df_total["sim_cfg"] == cfg) & (df_total["simulation_run"] == sim_run)]
        plan_demand = calculate_demand_percentage(curr_total, "low")
        plan_channels = g["channels"].astype(int).sum()
        plan_total_compute, plan_peak_compute, baseline = process_workflow_stats(base_dir, g)

        if plan_total_compute == 0:
            continue

        nodes = json.loads(g["nodes"].iloc[0])
        mean_node_flops, mean_node_bandwidth = get_compute_node_statisics(nodes)
        plan_statistics = {
            "cfg": cfg,
            "baselines": ','.join(str(b) for b in sorted(g['baseline'].unique())),  # TODO change to baseline ratio
            # "observation_plan_size": g["observation_size"].iloc[0],
            "large": g['large'].iloc[0],
            "medium": g['medium'].iloc[0],
            "small": g['small'].iloc[0],
            "demand_ratio": round(plan_demand, 2),
            "channels_ratio": round(plan_channels / (max_channels * len(g)), 2),
            "use_task_data": g["use_task_data"].iloc[0],
            "use_edge_data": g["use_edge_data"].iloc[0],
            # "data": g["data"].iloc[0],
            # "data_distribution": g["data_distribution"].iloc[0],
            "max_running_tasks": g["max_running_tasks"].iloc[0],
            "min_running_tasks": g["min_running_tasks"].iloc[0],
            "mean_running_tasks": g["mean_running_tasks"].iloc[0],
            "schedule_length": g["schedule_length"].iloc[0],
            "planning": planning,
            "computing_to_observation_length_ratio": (g["schedule_length"].iloc[0]) / g["total_obs_duration"].iloc[0],
            "max_ingest_flops": (g["max_ingest_demand"].iloc[0] * mean_node_flops / 1e15),
            "mean_ingest_flops": (g["mean_ingest_demand"].iloc[0] * mean_node_flops / 1e15),
            "mean_node_bandwidth": mean_node_bandwidth
        }
        plan_statistics.update(calculate_week_on_telescope_comparison_stats(plan_total_compute,
                                                                            plan_peak_compute))

        # "Schedule length from simulation is time-step dependent, need to multiply by the timestep to get seconds
        schedule_length_seconds = int(g["schedule_length"].iloc[0]) * timestep
        plan_statistics.update(calculate_average_flops_in_plan(plan_total_compute,
                                                               schedule_length_seconds,
                                                               mean_node_flops,
                                                               plan_statistics["mean_running_tasks"],
                                                               plan_statistics["max_running_tasks"]))

        usage.append(plan_statistics)
        count += 1

    # Fuzzy find data usage
    # data_usage_group = df_total.groupby(["sim_cfg", "planning"])

    return pd.DataFrame(usage)


def plot_plan_compostition(df_total):
    """
    Plot the composition of the plan over the duration of the simulation based on
    (cumulative) use of a telescope parameter over time

    Parameters
    ----------
    df_total

    Returns
    -------

    """


def setup_axes(axes: list):
    """
    Apply common axes settings so we have consistent presentation
    """
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(True, "major", "both", ls="-", color="black")
        ax.grid(True, "minor", "both", ls="--")

    return axes


import matplotlib

def plot_box_axis(usage: pd.DataFrame,
                      ax: matplotlib.axes,
                      xaxis: str = "computing_to_observation_length_ratio",
                      yaxis: str = "plan_average_compute_from_flops", use_legend=False, **kwargs):
    algorithms = kwargs.get('algorithms')
    markers = kwargs.get("markers", {'heft': 'o'})
    colors = kwargs.get('colors', {})
    labels = kwargs.get('labels', {'':''})
    positions = kwargs.get('positions', {'heft':1})
    data=[]
    labels = labels.keys()
    color = []
    pos = []
    for alg in algorithms:
        # data_points = len(usage[usage["planning"] == planning])
        result = usage[(usage["planning"] == alg)]
        data.append(result[yaxis].to_numpy())
        color.append(colors[alg])
        pos.append(positions[alg])
    # y = np.stack(data, axis=-1)
    sc = ax.boxplot(
        data,
        tick_labels=list(algorithms.values()),
        positions=pos,
        patch_artist=True,
        label=list(labels)[0]
    )

    for patch, c in zip(sc['boxes'], color):
        if c:
            patch.set_facecolor(c)
        else:
            patch.set_facecolor('none')


    for median in sc['medians']:
        median.set_color('black')

    return ax, sc



def plot_scatter_axis(usage: pd.DataFrame,
                      ax: matplotlib.axes,
                      xaxis: str = "computing_to_observation_length_ratio",
                      yaxis: str = "plan_average_compute_from_flops", use_legend=False, **kwargs):
    algorithms = kwargs.get('algorithms')
    markers = kwargs.get("markers", {'heft': 'o'})
    fill_plots = kwargs.get('fill', False)

    legend_elements = []

    # Dummy headings (invisible)
    legend_elements.append(plt.Line2D([0], [0], linestyle='none', label='Algorithms'))

    for algorithm, marker in markers.items():
        legend_elements.append(
            plt.Line2D([0], [0], marker=marker, color='black', linestyle='None', label=algorithms[algorithm]))

    # unique_io = usage['mean_node_bandwidth'].unique()
    # min_io = unique_io.min()
    # io_values = unique_io/min_io
    colors = kwargs.get('colors', {})

    # plot_io_variation = kwargs.get('all_io', False)
    io_idx = 0
    # for i, color in enumerate(colors):
    for planning in algorithms:
        # data_points = len(usage[usage["planning"] == planning])
        result = usage[(usage["planning"] == planning)]
        # result = result[result["mean_node_bandwidth"] == unique_io[i]]
        # io = round(unique_io[i]/min_io, 1)
        points = np.column_stack((result[xaxis].to_numpy(), result[yaxis].to_numpy()))
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        sc = ax.scatter(
            points[:, 0],
            points[:, 1],
            s=20,
            marker=markers[planning],
            color=colors[planning],
            label=algorithms[planning],
            edgecolors='black'
        )

        # Plot uses marker face color
        if fill_plots:
            ax.fill(hull_points[:, 0], hull_points[:, 1], color='lightblue', alpha=0.25)
    io_idx += 1

    if use_legend:
        legend = ax.legend(handles=legend_elements, alignment='left', loc='upper left', frameon=True, handlelength=2,
                           fontsize='small')
        for text in legend.get_texts():
            if text.get_text() in ['Algorithms', 'I/O']:
                text.set_weight('bold')
        #         text.set_ha("left")
    return ax, sc

def plot_ternary_axis(usage: pd.DataFrame,
                      ax: matplotlib.axes,
                      gs, fig,
                      difference: bool = True,
                      zvalue: str = "computing_to_observation_length_ratio", use_legend=False, **kwargs):

    algorithms = kwargs.get('algorithms')
    markers = kwargs.get("markers", {'default': 'o'})
    fill_plots = kwargs.get('fill', False)

    legend_elements = []

    # Dummy headings (invisible)
    legend_elements.append(plt.Line2D([0], [0], linestyle='none', label='Algorithms'))

    for algorithm, marker in markers.items():
        legend_elements.append(
            plt.Line2D([0], [0], marker=marker, color='black', linestyle='None', label=algorithms[algorithm]))

    io_idx = 0
    contour = True

    # top left
    ax1 = fig.add_subplot(gs[0:1, 0:1])
    # bottom left
    ax2 = fig.add_subplot(gs[1:2, 0:1])
    # Right
    ax3 = fig.add_subplot(gs[0:2,1:2])
    axes = [ax1, ax2, ax3]
    from scipy.interpolate import griddata
    if contour and difference:
        x_min = usage["small"].min()*100
        x_max = usage["small"].max()*100
        y_min = usage["medium"].min()*100
        y_max = usage["medium"].max()*100

        grid_x, grid_y = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100)
        )

        grids = {}  # store interpolated grids for each dataset
        for planning in algorithms:
            result = usage[usage["planning"] == planning]
            grid_z = griddata(
                (result["small"]*100, result["medium"]*100),
                result[zvalue],
                (grid_x, grid_y),
                method='linear'
            )
            grids[planning] = grid_z

        # Assume exactly two datasets for difference plot
        planning_keys = list(algorithms.keys())
        grid1 = grids[planning_keys[0]]
        grid2 = grids[planning_keys[1]]

        # Compute difference (grid2 - grid1)
        diff_grid = grid1 - grid2

        # Set symmetric color scale around zero
        max_abs = np.nanmax(np.abs(grid1))
        # max2 = np.nanmax(np.abs(grid2))
        # max_abs = max(max1, max2)


        sc = ax3.imshow(
            diff_grid,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            cmap="plasma",
            vmin=0,
            vmax=4,
            aspect="auto",
            label=planning_keys[0]
        )

        sc = ax2.imshow(
            grid2,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            cmap="plasma",
            vmin=0,
            vmax=4,
            aspect="auto",
            label=planning_keys[1]
        )

        sc = ax1.imshow(
            grid1,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            cmap="plasma",
            vmin=0,
            vmax=4,
            aspect="auto",
            label=planning_keys[1]
        )


        # Plot the 10% large line
        x, y = np.meshgrid(np.linspace(x_min, x_max, 10),
        np.linspace(y_min, y_max, 10))

        x = np.linspace(0, 100, 21)
        y = np.linspace(0, 100, 21)
        X, Y = np.meshgrid(x, y)
        Z = 100 - (X + Y)
        mask = (Z<=21) & (Z>=19)
        # mask =  (Z > 0.06) & (Z <0.1)
        ax3.plot(X[mask], Y[mask], color='black', linewidth=0.25, linestyle='--') # , s=10)

        for i, (xi, yi, zi) in enumerate(zip(X[mask], Y[mask], Z[mask])):
            if i == 9:
                ax1.text(xi + 0.00, yi + 0.00, f"{int(zi)}%", fontsize=4)

        mask = (Z <= 11) & (Z >= 9)
        ax3.plot(X[mask], Y[mask], color='black', linewidth=0.25, linestyle='--')  # , s=10)

        for i, (xi, yi, zi) in enumerate(zip(X[mask], Y[mask], Z[mask])):
            if i == 8:
                ax1.text(xi + 0.00, yi + 0.00, f"{int(zi)}%", fontsize=4)


    else:

        # Create two plot that plots the actual values.

        for planning in algorithms:
            result = usage[(usage["planning"] == planning)]
            sc = ax.scatter(
                result["small"], result["medium"], c=result[zvalue],
                s=20,
                marker=markers[planning],
                # color=colors[planning],
                label=algorithms[planning],
                edgecolors='black'
            )

    io_idx += 1

    if use_legend:
        legend = ax.legend(handles=legend_elements, alignment='left', loc='upper left', frameon=True, handlelength=2,
                           fontsize='small')
        for text in legend.get_texts():
            if text.get_text() in ['Algorithms', 'I/O']:
                text.set_weight('bold')
        #         text.set_ha("left")
    return axes, sc



def plot_histogram_axis(usage, ax, xaxis, **kwargs):
    plot_data = []
    algorithms = kwargs.get('algorithms')
    zorder = [2, 1]
    alpha = [0.5, 1]
    linewidth = [2, 1]
    labels = kwargs.get('labels')
    for i, planning in enumerate(algorithms):
        res = usage[(usage["planning"] == planning)]
        # plot_data.append(np.array(sorted(res[xaxis]), dtype='float').T)
        data = np.array(sorted(res[xaxis]), dtype='float').T
        sc = ax.hist(
            data,
            bins=np.arange(0.0, 4, 0.25),
            hatch=labels['hatch'][i],
            facecolor=labels['color'][i],
            label=labels['labels'][i],
            edgecolor='black',
            zorder=zorder[i],
            linewidth=linewidth[i],
            weights=np.ones(len(data)) / len(data),
            # edgecolor=labels['color'][i],
            # stacked=False,
            # fill=False,
            alpha=alpha[i]
        )

    return ax, sc


def create_figure(nrows, ncols, twocolumn=False, ):
    if twocolumn:
        fig = plt.figure(figsize=(6, 4), dpi=300)
    else:
        fig = plt.figure(figsize=(10 / 3, 3), dpi=300)
    right = 0.7 if twocolumn else 0.85
    gs = GridSpec(
        nrows, ncols, figure=fig, hspace=0.0, bottom=0.15, right=right, left=0.15,
    )  # , wspace=0.25) # , left=0.05, right=0.1, wspace=0.05)
    return fig, gs


def histogram_with_dataframe(usage, fig=None, gs=None, axis=None,
                             data=True,
                             data_distribution="edges",
                             xaxis="computing_to_observation_length_ratio",
                             yaxis="plan_average_compute_from_flops",
                             plot_type="hist", **kwargs):
    columns = kwargs.get("columns", 1)
    rows = kwargs.get("rows", 1)
    gs_position = kwargs.get("gs_position", (0, 0))
    two_column = kwargs.get('twocolumn', False)
    if not fig:
        fig, gs = create_figure(rows, columns, twocolumn=two_column)


def plot_with_dataframe(usage, fig=None, gs=None, axis=None,
                        use_task_data=True,
                        use_edge_data=False,
                        xaxis="computing_to_observation_length_ratio",
                        yaxis="plan_average_compute_from_flops",
                        plot_type="hist", **kwargs):
    """
    :param usage:
    :return:
    """
    label_map = {"batch": "blue", "heft": "red"}
    usage = usage[(usage["use_task_data"] == use_task_data) & (usage["use_edge_data"] == use_edge_data)]
    columns = kwargs.get("columns", 1)
    rows = kwargs.get("rows", 1)
    gs_position = kwargs.get("gs_position", (0, 0))
    k = kwargs.get('k')
    col = kwargs.get('col')
    row = kwargs.get('row')
    two_column = kwargs.get('twocolumn', False)
    if not fig:
        fig, gs = create_figure(rows, columns, twocolumn=two_column)
    if axis:
        ax = axis
    else:
        if k:
            num_cols = row + 1
            offset = (k - num_cols) // 2
            ax = fig.add_subplot(gs[row, col])
        if plot_type == "ternary":
            ax = None
        else:
            ax = fig.add_subplot(gs[*gs_position])
    if plot_type == "hist":
        ax, sc = plot_histogram_axis(usage, ax, xaxis, **kwargs)
    if plot_type == "scatter":
        # label_map = kwargs.get("labels", {"HEFT": "red"})
        # markers = kwargs.get("markers", 'o')
        ax, sc = plot_scatter_axis(usage, ax, xaxis, yaxis, **kwargs)
    if plot_type == "ternary":
        ax, sc = plot_ternary_axis(usage, ax, gs=gs, fig=fig, **kwargs)

    if plot_type == "box":
        ax, sc = plot_box_axis(usage, ax, xaxis, yaxis, **kwargs)

    if not isinstance(ax, list):
        ax.set_axisbelow(True)
        ax.grid(True, "major", "both", ls="--", color="grey")
        # ax.grid(True, "minor", "both", ls="--")

        ax.set_xlabel(xaxis)
        if plot_type == "hist":
            pass
        if plot_type == "scatter":
            ax.set_ylabel(f"{yaxis}")

    # Select data points from data.

    # ax.legend()
    return fig, gs, ax, sc


def plot_alternative_scatter(usage, data=True, data_distribution="edges", zoom=False):
    fig = plt.figure(figsize=(12, 12))
    # Show only the successful results for batch_planning
    success = True
    algorithms = ["batch", "heft"]  # ["BatchPlanning", "SHADOWPlanning"]
    colors = ["blue", "red"]
    from textwrap import wrap
    average_flops_low = 13.8
    y_max = 100

    fig.suptitle("Percentage of successful or failed \nplans across telescope demand")
    gs = GridSpec(
        8, 8, figure=fig, hspace=0.25, wspace=1
    )  # , wspace=0.25) # , left=0.05, right=0.1, wspace=0.05)
    ax1 = fig.add_subplot(gs[2:, 0:4])
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax2 = fig.add_subplot(gs[0:2, 0:4], sharex=ax1)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax3 = fig.add_subplot(gs[0:4, 4:])
    ax4 = fig.add_subplot(gs[5:, 4:])
    ax1, ax2, ax3, ax4 = setup_axes([ax1, ax2, ax3, ax4])
    ax1.set_axisbelow(True)
    ax1.grid(True, "major", "both", ls="-", color="black")
    ax1.grid(True, "minor", "both", ls="--")

    metric = "computing_to_observation_length_ratio"
    usage = usage[(usage["data"] == data) & (usage["data_distribution"] == data_distribution)]

    boxes = []
    plot_data = []
    for i, planning in enumerate(algorithms):
        res = usage[(usage["planning"] == planning)]
        plot_data.append(np.array(sorted(res[metric]), dtype='float').T)

    ### Plot histogram of each simulation and the resultant metric ###
    ### Metrics include: "schedule_length_ratio", "peak_compute", "observation_to_computing"

    ax1.hist(
        plot_data,
        bins=np.arange(1, 5, 0.25),
        # weights=np.ones_like(data) / len(data), # TODO Consider revisiting
        # hatch="xx",histtype='bar',
        facecolor=colors,
        edgecolor='black',
        stacked=True,
        alpha=0.7
    )
    # ax1.legend(algorithms)
    for i, planning in enumerate(algorithms):
        # data_points = len(usage[usage["planning"] == planning])
        result = usage[(usage["planning"] == planning)]

        demand = set(result["demand_ratio"])
        # for x in demand:
        #     boxes.append(np.array(result[result["demand_ratio"] == x]["relative_average_compute"]))

        # data = np.array(sorted(result["schedule_length_ratio"]), dtype='float').T
        ax2.scatter(
            result['computing_to_observation_length_ratio'].to_numpy(),
            result['plan_average_compute_from_flops'].to_numpy(),
            label=["Demand ratio"],
            marker='o',
            s=50,
            color=colors[i],
            edgecolors='black'
        )
        # This is plan relative compute
        # we could experiment with plotting each plan with the relative average compute against the relative
        # total FLOPS? This would show the relationship between achieved and expected compute
        ax4.scatter(
            result['relative_total_compute'].to_numpy(),
            result['relative_average_compute'].to_numpy(),
            marker='+',
            s=50,
            color=colors[i],
            edgecolors='black'
        )

        ax3.scatter(
            result['demand_ratio'].to_numpy(),
            result[metric].to_numpy(),
            color=colors[i]
        )

        # ax4.scatter(
        #     result['channels_ratio'].to_numpy(),
        #     result[metric].to_numpy(),
        #     color=colors[i]
        # )

    # AX1 LABELS
    ax1.set_ylabel("Number of simulations")
    ax1.set_xlabel("Schedule length ratio")
    ax1.set_xlim([0.5, 5])

    # AX2 LABELS
    ax2.set_ylabel("Petaflops")
    ax2.legend(algorithms)
    ax2.set_ylim([0, 15])
    ax2.plot([0.0, 5.0], [average_flops_low, average_flops_low], "--",
             color=" red",
             linewidth=3, zorder=-1)

    # AX3 & AX4 LABELS
    ax3.set_xlabel("Telescope Demand % over 7-days ()")
    ax3.set_ylabel("Schedule length ratio")
    ax3.legend(labels=algorithms)
    ax4.set_xlabel("Telescope Channel % over 7-days ()")
    # ax4.set_ylabel("Schedule length ratio")
    ax4.legend(labels=algorithms)

    plt.tight_layout(pad=3.0)

    x1, x2, y1, y2 = 0.25, 0.35, 0.9, 2.0
    show_zoom = False
    if show_zoom:
        axins = ax1.inset_axes(
            [0.5, 0.5, 0.47, 0.47],
            xlim=(x1, x2),
            ylim=(y1, y2),
            xticklabels=[],
            yticklabels=[],
        )
        result = usage[(usage["success"] == success) & (usage["planning"] == planning)]
        axins.scatter(
            np.array(result["demand_ratio"]),
            np.array(result["schedule_length_ratio"]),
            marker="o",
            color="grey",
        )
        result = usage[(usage["success"] != success) & (usage["planning"] == planning)]
        axins.scatter(
            np.array(result["demand_ratio"]),
            np.array(result["schedule_length_ratio"]),
            marker="+",
            color="red",
        )

        ax1.indicate_inset_zoom(axins, edgecolor="black")


def calculate_maximum_moving_average_for_observing_plan(observation_plan):
    """
    This returns a maximum moving average for the observing plan resource use

    For a sequence of observations, we can calculate the average resource use over the
    course of those observations. This average can give us a indicator of the utilisation
    of the telescope over those observations.

    If we want to determine if an observation plan has particularly high resource use all at
    once, we would expect that to have a higher max-moving average than an observing plan
    with the same number of high demand observations that are more distributed.

    Returns
    -------

    """
    y = np.array(observation_plan['instrument_demand'])
    x = np.array(observation_plan['start'])
    color_map = {"hpso01": "red", "hpso02a": "blue", "hpso02b": "yellow", "hpso04a": "green", "hpso05a": "orange"}
    dur = np.array(observation_plan['duration'])


def json_plan_to_dataframe(config_path: Path) -> pd.DataFrame:
    sim_cfg = json.load(fp)
    return pd.DataFrame(
        sim_cfg["instrument"]["telescope"]["observations"])


def get_observation_plans(df_total: pd.DataFrame, config_dir: Path) -> pd.DataFrame:
    """
    For each simulation config file, get the observation plan
    """

    plans = []
    for config in set(df_total['sim_cfg']):
        path = config_dir / config
        with path.open('r') as fp:
            sim_cfg = json.load(fp)
            observation_plan = pd.DataFrame(
                sim_cfg["instrument"]["telescope"]["observations"])
            observation_plan['config'] = config
            plans.append(observation_plan)
    return pd.concat(plans, ignore_index=True)


def plot_observation_plan(observation_plan: pd.DataFrame, demand: float):
    """
    Show the telescope usage of each observation across the simulation.

    Key columns of observation_plan are:
    - name
    - start
    - duratio
    - instrument_demand
    """
    fig, gs = create_figure(1, 1, False)
    # fig = plt.figure(figsize=(10/3, 3), dpi=300)
    # gs = GridSpec(
    #     1, 1, figure=fig
    # )  # , wspace=0.25) # , left=0.05, right=0.1, wspace=0.05)
    ax1 = fig.add_subplot(gs[0, 0])
    # ax1 = setup_axes([ax1])[-1]
    # Need to map width to the right
    y = np.array(observation_plan['instrument_demand'])
    x = np.array(observation_plan['start'])

    import matplotlib.colors as mcolors
    css_colors = list(mcolors.CSS4_COLORS.keys())
    color_map = {"hpso01": css_colors[5], "hpso02a": css_colors[12], "hpso02b": css_colors[19],
                 "hpso04a": css_colors[128]
        , "hpso05a": css_colors[80]}

    dur = np.array(observation_plan['duration'])
    observation_plan.loc[:, 'color'] = observation_plan.loc[:, 'type'].map(color_map)
    colors = np.array(observation_plan['color'])
    hpso = np.array(observation_plan['type'])
    plotted = set()
    for i, e in enumerate(x):
        if hpso[i] in plotted:
            ax1.broken_barh([(x[i], dur[i])], (0, y[i]), facecolors=colors[i])
        else:
            ax1.broken_barh([(x[i], dur[i])], (0, y[i]), facecolors=colors[i], label=hpso[i].upper())
            plotted.add(hpso[i])
    handles, labels = ax1.get_legend_handles_labels()
    order = np.argsort(labels)
    ax1.set_yticks([0, 64, 128, 256, 512])
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    def seconds_to_days(x, pos):
        return f"{int(x / 24 / 3600)}"  # e.g., 7200 -> 2.0

    # ax1.bar(x=x, height=y, width=dur, color=colors, edgecolor="black")
    ax1.xaxis.set_major_locator(MultipleLocator(86400))
    ax1.xaxis.set_major_formatter(FuncFormatter(seconds_to_days))
    ax1.set_xlabel('Time (Days)')
    ax1.set_ylabel("# Stations")
    ax1.set_ylim(0, 512)
    ax1.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    fig.suptitle(f"{demand}")


def select_n_configs_by_key(usage_summary: pd.DataFrame, key: str, value: object, count=2):
    """
    Select 'n' number of config files based on key and value pair. Useful to get subset information
    for more specific analysis or visualisiation.
    """
    demand_df = usage_summary[usage_summary[key] == value]
    cfgs = set(demand_df["cfg"])  #

    chosen_cfgs = random.sample(list(cfgs), count)

    return chosen_cfgs


def setup_parser():
    import argparse
    parser = argparse.ArgumentParser(Path(__file__).name)
    parser.add_argument("base_dir", help="The directories in which simulation HDF5 files are stored relative to base_dir.")
    parser.add_argument("--result_dirs", nargs="+", required=True, help="The directories in which simulation HDF5 files are stored relative to base_dir.")
    parser.add_argument("--total", help="(Path) Complete results stored in .csv file. If not provided, filename is generated.")
    parser.add_argument("--summary", help="(Path) Summary data and statistics based on --total .csv. If not provided, filename is generated.")
    parser.add_argument("--reprocess", default=False, action="store_true", help="Reprocess results into total or summary files.")
    parser.add_argument("--append", default=False, action="store_true",
                        help="Append results into total or summary files.")
    parser.add_argument("--algorithms", nargs='+', required=True, help="The algorithms for which we want to keep results data.")
    return parser.parse_args()


def gridspec_experiment():
    fig = plt.figure()
    gs = GridSpec(2, 2, fig)
    ax1 = fig.add_subplot(gs[0:1, 0:1])
    ax1.set_xlabel("Top Left")
    ax2 = fig.add_subplot(gs[0:1, 1:2])
    ax2.set_xlabel("TopRight")
    ax3 = fig.add_subplot(gs[1:2, 0:1])
    ax3.set_xlabel("BottomLeft")
    ax4 = fig.add_subplot(gs[1:2, 1:2])
    ax4.set_xlabel("BottomRight")


def generate_total_dataframe(
        df_total_path: Path,
        result_paths: list[Path],
        base_dir: Path,
        algorithms: list,
        reprocess: bool,
        append: bool,
        debug=False):
    """
    Produce the full result data frame

    :param df_total_path:
    :param result_paths:
    :param reprocess:
    :return:
    """
    df_total = None
    simulation_summaries = {}
    fetch_summaries_only = False  # Make this a CLI option
    if not df_total_path.exists() or reprocess or debug or append:
        # df_total_path.unlink(missing_ok=True)
        for simulation_batch in extract_simulations_from_hdf5(result_paths, verbose=True):
            if not simulation_batch:
                print(f"No simulation batch for {result_paths}")
                continue
            for simulation, dtype in simulation_batch.items():
                cfg = dtype['sim']['config'].iloc[0]
                simulation_summaries[cfg] = dtype["summary"].to_csv()
            if fetch_summaries_only:
                continue
            df_total = collate_simulation_results(base_dir, simulation_batch)
            df_total = convert_categorical_ints_to_str(df_total)
            sbef = len(df_total)
            logging.info("Filtering data with the following algorithms: %s", algorithms)
            logging.info("Size of data frame before filtering: %d", len(df_total))
            df_total = df_total[df_total['planning'].isin(algorithms)]
            saft = len(df_total)
            perc_diff = ((sbef-saft)/sbef*100)
            logging.info("Size of data frame after filtering: %d (%d %% reduction)", saft, perc_diff)
            if debug:
                continue
            else:
                if df_total_path.exists():
                    try:
                        with df_total_path.open("a") as fp:
                            df_total.to_csv(fp, mode='a', header=False)
                    except pd.errors.ParserError:
                        print(f"Simulation batch caused issues writing to file: {simulation_batch}")
                else:
                    try:
                        with df_total_path.open("w") as fp:
                            df_total.to_csv(fp)
                    except pd.errors.ParserError:
                        print(f"Simulation batch caused issues writing to file: {simulation_batch}")
            simulation_batch = {}  # "Memory management" in Python
        with open("simulation_summaries.json", 'w') as fp:
            json.dump(simulation_summaries, fp, indent=2)

    return df_total, simulation_summaries


def plot_flops_vs_demand_low(usage_summary_dataframe, algorithm):

    # TODO Modify this so that we produce boxplots of achieved plots.
    node_flops, memory_bandwidth = get_compute_node_statisics(json.loads(df_total["nodes"].iloc[0]))

    usage_summary_dataframe["average_plus_ingest"] = usage_summary_dataframe["plan_average_compute_from_nodes"] + (
            (node_flops * LOW_REALTIME_RESOURCES) / 1e15)

    # TODO compare HEFT and BatcH
    fig, gs, ax, sc = plot_with_dataframe(usage=usage_summary_dataframe, use_task_data=False, use_edge_data=False,
                                          plot_type="box",
                                          xaxis="algorithm",
                                          yaxis="average_plus_ingest",
                                          title="demonstrate averate compute from nodes",
                                          algorithms={'batch': 'Batch', 'heft': 'HEFT'},
                                          colors={'heft': 'lightblue', 'batch':'lightblue'},
                                          labels={"Ave. FLOPS": "lightblue"}, twocolumn=True, positions={'heft': 1,'batch': 2})

    usage_summary_dataframe["peak_plus_ingest"] = usage_summary_dataframe["plan_peak_compute_from_nodes"] + (
                (node_flops * LOW_REALTIME_RESOURCES) / 1e15)
    fig, gs, ax, sc = plot_with_dataframe(usage=usage_summary_dataframe,
                                          axis=ax, fig=fig, gs=gs,
                                          use_task_data=False, use_edge_data=False, plot_type="box",
                                          xaxis="demand_ratio",
                                          yaxis="peak_plus_ingest", title="demonstrate averate compute from nodes",
                                          algorithms={'batch': 'Batch', 'heft': 'HEFT'},
                                          colors={'heft': 'red', 'batch':'red'},
                                          labels={"Max. FLOPS": "red"}, twocolumn=True, positions={'heft': 1,'batch': 2})
    # fig, gs, ax, sc = plot_with_dataframe(usage=usage_summary_dataframe,
    #                                       axis=ax, fig=fig, gs=gs,
    #                                       use_task_data=False, use_edge_data=False, plot_type="box",
    #                                       xaxis="demand_ratio",
    #                                       yaxis="mean_ingest_flops", title="demonstrate averate compute from nodes",
    #                                       algorithms={'batch': 'Batch', 'heft': 'HEFT'},
    #                                       colors={'heft': 'blue', 'batch': 'red'},
    #                                       markers={'heft':'x'}, twocolumn=True,
    #                                       positions={'heft': 1,'batch': 2})
    fig, gs, ax, sc = plot_with_dataframe(usage=usage_summary_dataframe,
                                          axis=ax, fig=fig, gs=gs,
                                          use_task_data=False, use_edge_data=False, plot_type="box",
                                          xaxis="demand_ratio",
                                          yaxis="max_ingest_flops", title="demonstrate averate compute from nodes",
                                          algorithms={'batch': 'Batch', 'heft': 'HEFT'},
                                          colors={'heft': 'pink', 'batch': 'pink'},
                                          markers={'heft':'x'}, twocolumn=True,
                                          positions={'heft': 1,'batch': 2},
                                          labels={"Max ingest FLOPS": "lightblue"})
    ax.legend(title="Per-Observing plan:", bbox_to_anchor=(1, 0.7))
    ax.set_ylim((0, 11))
    ax.set_ylabel("PetaFLOPs 'acheived'")
    # ax.set_xlabel("Demand Ratio")  # \n(# stations used across the observing plan / Total possible number of stations)")
    ax.set_xlim((0.0, 3))
    ax.plot([0.0, 5], [LOW_SDP_AVERAGE_COMPUTE_FLOPS_UPDATED, LOW_SDP_AVERAGE_COMPUTE_FLOPS_UPDATED],
            color="red", linestyle='--', linewidth=3, zorder=-1)  # , text="Updated estimated for SDP maximum compute")
    from matplotlib.patches import FancyArrowPatch
    arr = FancyArrowPatch((.4, 11), (.3, 10),
                          arrowstyle='->,head_width=.15', mutation_scale=20)
    # ax.add_patch(arr)
    fig.text(0.72, .73, "SDP Total Compute \n Adjusted Estimates\n")
    reserved_ingest = ((node_flops * LOW_REALTIME_RESOURCES) / 1e15)
    ax.plot([0.0, 5], [reserved_ingest, reserved_ingest],
            color="grey", linestyle='--', linewidth=3, zorder=-1)
    ax.fill_between([0, 3], y1=0, y2=reserved_ingest, color='grey', alpha=0.3, zorder=-1)
    fig.text(0.72, .2, "SDP Ingest\n Adjusted Estimates\n")
    fig.text(0.29, .16, "Ingest   reserved  resources\n")
    if SAVE_PLOTS:
        plt.savefig("FLOPS.png", dpi=fig.dpi)


def plot_histogram_observing_computing_ratio(usage_summary_dataframe):
    category_counts = usage_summary_dataframe['cfg'].value_counts()
    categories_to_remove = category_counts[category_counts < 6].index
    # Filter the DataFrame to keep only rows where the 'category' is NOT in the list of categories to remove
    usage_summary_dataframe = usage_summary_dataframe[~usage_summary_dataframe['cfg'].isin(categories_to_remove)]

    fig, gs, ax1, sc = plot_with_dataframe(usage=usage_summary_dataframe, use_task_data=False, use_edge_data=False,
                                           plot_type="hist",
                                           algorithms=['batch', 'heft'],
                                           labels={'labels': ['Batch', 'HEFT'], 'hatch': ['x', ''],
                                                       'color': ['silver', 'slateblue']}, rows=3, columns=1,
                                           gs_position=(0, 0))
    ax1.set_xlabel("")
    ax1.legend()
    fig, gs2, ax2, sc = plot_with_dataframe(usage=usage_summary_dataframe, fig=fig, gs=gs, use_task_data=True,
                                            use_edge_data=False, plot_type="hist",
                                            algorithms=['batch', 'heft'],
                                            labels={'labels': ['Batch', 'HEFT'], 'hatch': ['x', ''],
                                                        'color': ['silver', 'slateblue']}, rows=3, columns=1,
                                            gs_position=(1, 0))
    ax2.set_xlabel("")
    # ax1.set_title("Without edge data")

    fig, gs, ax3, sc = plot_with_dataframe(usage=usage_summary_dataframe, fig=fig, gs=gs2, use_task_data=True,
                                           use_edge_data=True, plot_type="hist",
                                           algorithms=['batch', 'heft'],
                                           labels={'labels': ['Batch', 'HEFT'], 'hatch': ['x', ''],
                                                       'color': ['silver', 'slateblue']}, rows=3, columns=1,
                                           gs_position=(2, 0))
    # ax2.set_title("With edge data")
    ax3.set_xlabel("")

    handles, labels = [], []
    for ax in [ax1]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    from collections import OrderedDict

    unique = OrderedDict(zip(labels, handles))

    # Establish limits based on maximum of two axes
    ax1_lim = ax1.get_ylim()
    ax2_lim = ax2.get_ylim()
    ax3_lim = ax3.get_ylim()
    lim = max([ax1_lim[1], ax2_lim[1], ax3_lim[1]])
    from matplotlib.ticker import PercentFormatter
    ax1.yaxis.set_major_formatter(PercentFormatter(1))
    ax2.yaxis.set_major_formatter(PercentFormatter(1))
    ax3.yaxis.set_major_formatter(PercentFormatter(1))
    ax1.tick_params(labelbottom=False)
    ax2.tick_params(labelbottom=False)
    ax1.set_ylim([0, lim])
    ax2.set_ylim([0, lim])
    #
    ax3.set_ylim([0, lim])
    ax1.set_xlim([0, 4])
    ax2.set_xlim([0, 4])
    ax3.set_xlim([0, 4])

    fig.supylabel("# of Simulations", fontsize=8)
    fig.supxlabel("Computing time to observing time ratio", fontsize=8)
    fig.text(0.9, 0.75, "(a)")
    fig.text(0.9, 0.5, "(b)")
    fig.text(0.9, 0.25, "(c)")

    if SAVE_PLOTS:
        fig.savefig("Histogram-with-all.png", dpi=fig.dpi)
    # fig3.legend(unique.values(), unique.keys(), handleheight=3, handlelength=3,fontsize='small', bbox_to_anchor=(0.85, 0.88), framealpha=1.0, edgecolor='white')
    # fig.legend()


def plot_demand_vs_observation_ratio_scatter(usage_summary_dataframe):
    import math
    # unique_baselines = usage_summary_dataframe["observation_plan_size"].unique()
    # nbaselines = len(unique_baselines)
    # k = math.ceil((-1 + math.sqrt(1 + 8 * nbaselines)) / 2)

    fig = plt.figure(figsize=(6, 4), dpi=300)
    gs = GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.2, bottom=0.10, right=0.85, left=0.1, )
    # )  # k rows, k columns to be safe
    plot_index = 0
    handles = []
    labels = []
    handles_labels_collected = False  # Flag to only collect once
    for row in range(1):
        # num_cols = row + 1  # increasing number of columns per row
        # col_offset = (k-num_cols) // 2
        # for col in range(num_cols):
        #     if plot_index >= nbaselines:
        #         break
        # baselines = unique_baselines[plot_index]
        # usage = usage_summary_dataframe[usage_summary_dataframe["observation_plan_size"] == baselines]
        usage = usage_summary_dataframe
        fig, gs, ax, sc = plot_with_dataframe(
            usage=usage, fig=fig, gs=gs, use_task_data=True, use_edge_data=True, plot_type="ternary",
            xaxis="demand_ratio", yaxis='computing_to_observation_length_ratio',
            algorithms={'batch': "Batch", 'heft': "HEFT"}, colors={'batch': 'red', 'heft': 'blue'},
            # , 2.0: 'slateblue', 5.0: 'lightsalmon' },
            markers={"batch": 'v', "heft": 'o'}, fill=False, gs_position=(row, 0)
        )
    labels = ['Batch', "HEFT", f"Δ (Batch − HEFT)"]
    for i,a in enumerate(ax):
        a.set_ylim([0.0, 100.0])
        a.set_xlim([0.0, 100.0])

        a.set_title(labels[i])
        plot_index += 1
        if not handles_labels_collected:
            handles, l = a.get_legend_handles_labels()
            handles_labels_collected = True

    # fig.colorbar(sc, ax=ax[0:2])
    fig.colorbar(sc, ax=ax[2], label='Computing to observing time ratio')# , label=labels[2])

    fig.supxlabel("% Small")
    fig.supylabel("% Medium")

    if SAVE_PLOTS:
        fig.savefig("scatter-ratio-multibaseline.png", dpi=fig.dpi)


SAVE_PLOTS = True

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    args = setup_parser()

    # Construct paths for processing
    base_dir = Path(args.base_dir)
    LOGGER.info("Result base directory: %s", base_dir)
    result_paths = []
    for rd in args.result_dirs:
        tmp = base_dir / rd
        if not tmp.exists():
            LOGGER.info("%s/%s does not exist!", base_dir, rd)
        else:
            LOGGER.info("Adding %s/%s to result paths", base_dir, rd)

        result_paths.append(tmp)

    fdate = datetime.date.today().strftime("%Y-%m-%d")
    if args.total:
        df_total_path = Path(args.total)
    else:
        df_total_path = Path(f"total_{fdate}.csv")
    LOGGER.info("Storing total results data in %s ", df_total_path)

    if args.summary:
        df_summary = Path(args.summary)
    else:
        df_summary = Path(f"summary_{fdate}.csv")
    LOGGER.info("Storing summary results data in %s ", df_summary)

    _, simulation_summaries = generate_total_dataframe(df_total_path,
                                                       result_paths,
                                                       base_dir,
                                                       algorithms=args.algorithms,
                                                       reprocess=args.reprocess,
                                                       append=args.append,
                                                       debug=False)
    # if df_total is None:
    df_total = pd.read_csv(df_total_path)
    if not simulation_summaries:
        with open("simulation_summaries.json") as fp:
            simulation_summaries = json.load(fp)

    if not df_summary.exists():
        usage_summary_dataframe = produce_summary_dataframe(df_total, base_dir)
        with df_summary.open("w") as fp:
            usage_summary_dataframe.to_csv(fp)
    else:
        usage_summary_dataframe = pd.read_csv(df_summary)

    ################################################################################
    ######                          MAKE PLOTS
    #################################################################################

    plot_histogram_observing_computing_ratio(usage_summary_dataframe)
    plot_demand_vs_observation_ratio_scatter(usage_summary_dataframe)
    plot_flops_vs_demand_low(usage_summary_dataframe, 'heft')

    # observation_plans = get_observation_plans(df_total=df_total,
    #                                           config_dir=result_path.parent)

    # SHOW CONFIGS

    # cfgs = select_n_configs_by_key(usage_summary_dataframe, "demand_ratio", 0.14, count=1)
    # plan = observation_plans[(
    #         observation_plans["config"] == cfgs[0])]
    # plot_observation_plan(plan, 0.14)

    # s = simulation_summaries[cfgs[0]]
    # create_simulation_schedule_map(pd.read_csv(StringIO(s)))

    # cfgs = select_n_configs_by_key(usage_summary_dataframe, "demand_ratio", 0.33, count=1)
    # plan = observation_plans[(
    #         observation_plans["config"] == cfgs[0])]
    # plot_observation_plan(plan, 0.33)
    #
    # s = simulation_summaries[cfgs[0]]
    # create_simulation_schedule_map(pd.read_csv(StringIO(s)))

    plt.show()
