#!/usr/bin/env python
# coding: utf-8
import datetime
import logging
import random
import shutil
import sys
import numpy as np
# import seaborn as sns
import tables
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
from skaworkflows.observation.statistics import observation_weighting
from skaworkflows.utils import create_observations_from_config

###########################################################
# PLOTTING PARARMETERS
###########################################################

# Setup all the visualisation nicities
rcParams["text.usetex"] = True
# rcParams["font.family"] = "serif"
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

DATE = datetime.datetime.now().strftime("%Y-%m-%d")

# Constants from updated SDP Cost Estimates (Alexander 2016)
TOTAL_COMPUTE_LOW_HPSOS_FLOPS = 1.2e24
LOW_HPSO_DURATION_YEARS = 2.8
MAXIMAL_OBSERVATION_COMPUTE_FLOPS = 1.5e21
SDP_AVERAGE_COMPUTE_FLOPS = 13.8e15
LOW_SDP_AVERAGE_COMPUTE_FLOPS_UPDATED = 9.623
LOW_REALTIME_RESOURCES = 164
MID_REALTIME_RESOURCES = 281

# ICAL = 'ICAL'                 # Produce calibration solutions using iterative self-calibration
# DPrepA = 'DPrepA'             # Produce continuum Taylor term images in Stokes I
# DPrepA_Image = 'DPrepA_Image' # Produce continuum Taylor term images in Stokes I as CASA does in images
# DPrepB = 'DPrepB'             # Produce coarse continuum image cubes in I,Q,U,V (with Nf_out channels)
# DPrepC = 'DPrepC'             # Produce fine spectral resolution image cubes un I,Q,U,V (with Nf_out channels)
# DPrepD = 'DPrepD'             # Produce calibrated, averaged (In time and freq) visibility data


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
                    try:
                        store = pd.HDFStore(str(p))
                    except tables.exceptions.HDF5ExtError:
                        continue
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
# 361c120c73b4441ea5d1962099a73790: 3.85, 0.06/0.11/0.83
# 4eb91e4f8771438ead59dfa8822bd957: 3.85. 0.02/0.37/0.61
def get_observation_plan_size(df):
    """
    Determine if the dataframe demand tuples match the different 'size' types above.

    We use the isdisjoint to determine if there are _any_ examples of the different tuples as
    defined above.

    :param df:
    :return:
    """
    counts = {"large": 0, "medium": 0, "small": 0}
    for _, row in df.iterrows():

        observation = row["observation"]
        pair = (row["baseline"], row["demand"])

        # Explicit small observations
        if 'hpso04a' in observation or 'hpso05a' in observation:
            counts["small"] += 1

        # Pair-based classification
        elif pair in SKALOW_LARGE_PAIRS:
            counts["large"] += 1

        elif pair in SKALOW_MED_PAIRS:
            counts["medium"] += 1

        else:
            counts["small"] += 1

    total = len(df)

    for key, value in counts.items():
        counts[key] = round(value / total, 2)

    return counts

def calculate_observation_durations(df_tel):
    # Collect all start/finish times for observations
    obs_durations = []
    intervals = []
    for obs in set(df_tel["observation"]):
        df_obs = df_tel[df_tel["observation"] == obs]
        start_time = df_obs[df_obs["event"] == "started"]["time"].iloc[0]
        finish_time = df_obs[df_obs["event"] == "finished"]["time"].iloc[0]
        intervals.append((start_time, finish_time))

    intervals.sort(key=lambda x: x[0])

    # Merge overlapping intervals
    merged_intervals = []
    if intervals:
        merged_intervals.append(intervals[0])
        for current_start, current_end in intervals[1:]:
            last_start, last_end = merged_intervals[-1]
            if current_start <= last_end:
                # If overlap, merge
                merged_intervals[-1] = (last_start, max(last_end, current_end))
            else:
                # if no overlap, add new interval
                merged_intervals.append((current_start, current_end))

    # Calculate durations from merged intervals
    for start, end in merged_intervals:
        obs_length = end - start
        obs_durations.append(obs_length)

    return obs_durations

def collate_simulation_results(parent_dir: Path, simulations: dict):
    # TODO consider applying timesteps to everything here so we don't have to later
    df_total = pd.DataFrame()
    processed = []
    for simulation, dtype in simulations.items():
        logging.info("Collating simulation results for %s", simulation)
        df = dtype["summary"]
        df_tel = df[(df["actor"] == "instrument")]
        obs_durations = calculate_observation_durations(df_tel)


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
        obs_plan_id = cfg["instrument"]["telescope"]["obs_plan_id"]
        resources = cfg['cluster']['system']['resources']
        str_resources = set([json.dumps(v) for v in resources.values()])
        obs_plan = create_observations_from_config(cfg)
        weighting = observation_weighting(obs_plan)
        nodes = [json.loads(s) for s in str_resources]

        parameters = (
            pd.DataFrame.from_dict(pipelines, orient="index")
            .reset_index()
            .rename(columns={"index": "observation"})
        )
        observations = pipelines.keys()

        # print(f"{parameters['workflow']}")
        parameters["timestep"] = timestep
        parameters["obs_plan_id"] = obs_plan_id
        parameters["weighting"] = weighting

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
    total_data = 0
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
            data = sum(total_workflow_df["total_data"]) * duration
            # This would be the compute for a given workflow
        else:
            # Pulsar is entire cost for the whole workflow
            compute = total_workflow_df.iloc[0]['total_cost'] * duration / (10 ** 15)
            data=0

        total_compute += compute
        total_data += data
        peak_compute = max(max_compute, compute)

    return total_compute, peak_compute, total_data,baseline


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
        plan_total_compute, plan_peak_compute,total_data, baseline = process_workflow_stats(base_dir, g)

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
            "plan_weighting": g["weighting"].iloc[0],
            "plan_id": g["obs_plan_id"].iloc[0],

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
            "mean_node_bandwidth": mean_node_bandwidth,
            "plan_total_data": total_data
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
        ax.grid(False, "major", "both", ls="-", color="black")
        ax.grid(False, "minor", "both", ls="--")

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
        label=list(labels)[0],
        vert=False,
        widths=[0.4,0.4]
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
                      yaxis: str = "plan_average_compute_from_flops", fig=None, **kwargs):
    algorithms = kwargs.get('algorithms')
    markers = kwargs.get("markers", {'heft': 'o'})
    fill_plots = kwargs.get('fill', False)
    draw_connection=kwargs.get('draw_connection', False)

    legend_elements = []

    # Dummy headings (invisible)
    legend_elements.append(plt.Line2D([0], [0], linestyle='none', label='Algorithms'))

    for algorithm, marker in markers.items():
        legend_elements.append(
            plt.Line2D([0], [0], marker=marker, color='black', linestyle='None', label=algorithms[algorithm]))

    use_legend = True
    # unique_io = usage['mean_node_bandwidth'].unique()
    # min_io = unique_io.min()
    # io_values = unique_io/min_io
    colors = kwargs.get('colors', {})

    # plot_io_variation = kwargs.get('all_io', False)
    io_idx = 0
    # for i, color in enumerate(colors):
    algs = list(algorithms.keys())
    variance=True
    for planning in algorithms:
        # data_points = len(usage[usage["planning"] == planning])
        result = usage[(usage["planning"] == planning)]
        if variance:
            points = np.column_stack((result[xaxis].to_numpy(), result[yaxis].to_numpy()))
            # grouped_df = result.groupby(xaxis)[yaxis]

            # x=result['plan_total_compute']
            y=result[yaxis]
            if xaxis == 'plan_weighting':
                color = result['plan_total_compute_flops']
            else:
                color=(1-result['small'])*100
            sc = ax.scatter(
                result[xaxis], y,  c=color, marker=markers[planning], label=algorithms[planning], cmap='viridis'
            )

    if use_legend:
        legend = ax.legend(handles=legend_elements, alignment='left', loc='upper left', frameon=True, handlelength=2,
                           fontsize='small')
        for text in legend.get_texts():
            if text.get_text() in ['Algorithms', 'I/O']:
                text.set_weight('bold')

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


        sc_diff = ax3.imshow(
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
            cmap="Reds",
            vmin=0,
            vmax=4,
            aspect="auto",
            label=planning_keys[1]
        )

        sc = ax1.imshow(
            grid1,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            cmap="Reds",
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
                ax3.text(xi + 0.00, yi + 0.00, f"{int(zi)}%", fontsize=4,)

        mask = (Z <= 11) & (Z >= 9)
        ax3.plot(X[mask], Y[mask], color='black', linewidth=0.25, linestyle='--')  # , s=10)

        for i, (xi, yi, zi) in enumerate(zip(X[mask], Y[mask], Z[mask])):
            if i == 8:
                ax3.text(xi + 0.00, yi + 0.00, f"{int(zi)}%", fontsize=4)


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
    return axes, sc, sc_diff



def plot_histogram_axis(usage, ax, xaxis, **kwargs):
    plot_data = []
    algorithms = kwargs.get('algorithms')
    zorder = [2, 1]
    alpha = [1, 1]
    linewidth = [1, 1]
    labels = kwargs.get('labels')
    for i, planning in enumerate(algorithms):
        res = usage[(usage["planning"] == planning)]
        # plot_data.append(np.array(sorted(res[xaxis]), dtype='float').T)
        data = np.array(sorted(res[xaxis]), dtype='float').T

        weights = np.ones(len(data))
        if i == 1:
            weights = -weights
        sc = ax.hist(
            data,
            bins=np.arange(0.0, 3, 0.2),
            # hatch=labels['hatch'][i],
            facecolor=labels['color'][i],
            label=labels['labels'][i],
            edgecolor='black',
            # zorder=zorder[i],
            linewidth=linewidth[i],
            weights=weights,
            # edgecolor=labels['color'][i],
            # stacked=False,
            # fill=False,
            alpha=alpha[i]
        )

    return ax, sc


def create_figure(nrows, ncols, twocolumn=False, large_legend=False, split_plot=False):
    if twocolumn:
        fig = plt.figure(figsize=(6, 4), dpi=300)
    else:
            fig = plt.figure(figsize=(10 / 3, 3), dpi=300)
    #
    if large_legend:
        # right = 0.5 #if twocolumn else 0.85

        gs = GridSpec(
            nrows, ncols, figure=fig, hspace=0.0, bottom=0.25,
            left=0.2, top=0.8
        )  # , wspace=0.25) # , left=0.05, right=0.1, wspace=0.05
        # gs = GridSpec(nrows, ncols, figure=fig)
    elif split_plot:
        right = 0.7 if twocolumn else 0.85
        gs = GridSpec(
            nrows, ncols, figure=fig, hspace=0.30, bottom=0.15, right=right, left=0.15,
        )  # , wspace=0.25) # , left=0.05, right=0.1, wspace=0.05)
    else:
        right = 0.7 if twocolumn else 0.90
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
    large_legend = kwargs.get('large_legend', False)
    ax_diff = kwargs.get('ax_diff', False)
    sc, sc_diff = None, None
    if not fig:
        fig, gs = create_figure(rows, columns, twocolumn=two_column,
                                large_legend=large_legend)
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
        ax, sc = plot_scatter_axis(usage, ax, xaxis, yaxis, **kwargs, fig=fig)
    if plot_type == "ternary":
        ax, sc, sc_diff = plot_ternary_axis(usage, ax, gs=gs, fig=fig, **kwargs)

    if plot_type == "box":
        ax, sc = plot_box_axis(usage, ax, xaxis, yaxis, **kwargs)

    if not isinstance(ax, list):
        ax.set_axisbelow(True)
        # ax.grid(False, "major", "both", ls="--", color="grey")
        # ax.grid(True, "minor", "both", ls="--")

        ax.set_xlabel(xaxis)
        if plot_type == "hist":
            pass
        if plot_type == "scatter":
            ax.set_ylabel(f"{yaxis}")
    # else:
    #     for a in ax:
    #         # a.grid(False, "major", "both", ls="--", color="grey")

    # Select data points from data.

    # ax.legend()
    return fig, gs, ax, sc, sc_diff



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
    ax1.set_ylabel("Stations")
    ax1.set_ylim(0, 512)
    ax1.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    fig.suptitle(f"{demand}")


def select_n_configs_by_key(usage_summary: pd.DataFrame, count=2):
    """
    Select 'n' number of config files based on key and value pair. Useful to get subset information
    for more specific analysis or visualisiation.
    """
    cfgs = set(usage_summary["cfg"])  #
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
    parser.add_argument("--experiment_type", help="Series, Concurrent or something else - string definition")
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
        LOGGER.info("Creating new total summary file: %s", str(df_total_path))
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
        with open(f"simulation_summaries_{DATE}.json", 'w') as fp:
            json.dump(simulation_summaries, fp, indent=2)
    else:
        LOGGER.info("Using existing total data: %s", str(df_total_path))

    return df_total, simulation_summaries


def plot_flops_vs_demand_low(usage_summary_dataframe, algorithm, experiment_type):

    # TODO Modify this so that we produce boxplots of achieved plots.
    node_flops, memory_bandwidth = get_compute_node_statisics(json.loads(df_total["nodes"].iloc[0]))

    usage_summary_dataframe["average_plus_ingest"] = usage_summary_dataframe["plan_average_compute_from_nodes"] + (
            (node_flops * LOW_REALTIME_RESOURCES) / 1e15)

    # TODO compare HEFT and BatcH
    fig, gs, ax, sc, sc_diff = plot_with_dataframe(usage=usage_summary_dataframe, use_task_data=True, use_edge_data=True,
                                          plot_type="box",
                                          xaxis="algorithm",
                                          yaxis="average_plus_ingest",
                                          title="demonstrate averate compute from nodes",
                                          algorithms={'batch': 'Batch', 'heft': 'HEFT'},
                                          colors={'heft': 'lightblue', 'batch':'lightblue'},
                                          labels={"$\\overline{F}$":
                                                      "lightblue"},
                                          positions={'heft': 1,'batch': 2},
                                          two_column=False,
                                          large_legend=True)
    #
    usage_summary_dataframe["peak_plus_ingest"] = usage_summary_dataframe["plan_peak_compute_from_nodes"] + (
                (node_flops * LOW_REALTIME_RESOURCES) / 1e15)
    fig, gs, ax, sc , sc_diff= plot_with_dataframe(usage=usage_summary_dataframe,
                                          axis=ax, fig=fig, gs=gs,
                                          use_task_data=True, use_edge_data=True, plot_type="box",
                                          xaxis="demand_ratio",
                                          yaxis="peak_plus_ingest", title="demonstrate averate compute from nodes",
                                          algorithms={'batch': 'Batch', 'heft': 'HEFT'},
                                          colors={'heft': 'red', 'batch':'red'},
                                          labels={"$max(F)$": "red"},
                                          positions={'heft': 1,'batch': 2})
    # # fig, gs, ax, sc = plot_with_dataframe(usage=usage_summary_dataframe,
    # #                                       axis=ax, fig=fig, gs=gs,
    # #                                       use_task_data=False, use_edge_data=False, plot_type="box",
    # #                                       xaxis="demand_ratio",
    # #                                       yaxis="mean_ingest_flops", title="demonstrate averate compute from nodes",
    # #                                       algorithms={'batch': 'Batch', 'heft': 'HEFT'},
    # #                                       colors={'heft': 'blue', 'batch': 'red'},
    # #                                       markers={'heft':'x'}, twocolumn=True,
    # #                                       positions={'heft': 1,'batch': 2})
    fig, gs, ax, sc , sc_diff = plot_with_dataframe(usage=usage_summary_dataframe,
                                          axis=ax, fig=fig, gs=gs,
                                          use_task_data=True, use_edge_data=True, plot_type="box",
                                          xaxis="demand_ratio",
                                          yaxis="max_ingest_flops", title="demonstrate averate compute from nodes",
                                          algorithms={'batch': 'Batch', 'heft': 'HEFT'},
                                          colors={'heft': 'pink', 'batch': 'pink'},
                                          markers={'heft':'x'},
                                          positions={'heft': 1,'batch': 2},
                                          labels={"$max(F_I)$": "lightblue"})
    ax.legend(title="", bbox_to_anchor=(1.05, -.2), ncol=3)
    ax.set_xlim((0, 11))
    ax.set_xlabel("PetaFLOPs 'acheived'")
    ax.set_ylabel("Scheduling method")
    # ax.set_xlabel("Demand Ratio")  # \n(# stations used across the observing plan / Total possible number of stations)")
    ax.set_ylim((0.0, 3))
    ax.axvline(x=LOW_SDP_AVERAGE_COMPUTE_FLOPS_UPDATED,
            color="red", linestyle='-', linewidth=1, zorder=-1)  # ,
    # text="Updated estimated for SDP maximum compute"
    from matplotlib.patches import FancyArrowPatch
    arr = FancyArrowPatch((.4, 11), (.3, 10),
                          arrowstyle='->,head_width=.15', mutation_scale=20)
    ax.add_patch(arr)
    fig.text(0.68, .73, "Max. SDP\n Ave. Compute*\n\n")
    reserved_ingest = ((node_flops * LOW_REALTIME_RESOURCES) / 1e15)
    ax.axvline(x=reserved_ingest,
            color="grey", linestyle='--', linewidth=1, zorder=-1)
    ax.fill_between([0, reserved_ingest], y1=0, y2=3, color='grey', alpha=0.3, zorder=-1)
    fig.text(0.2, .82, "SDP Ingest*")
    fig.text(0.23, .28, "Ingest   reserved\n", rotation=270)
    if SAVE_PLOTS:
        plt.savefig(f"FLOPS-{experiment_type}-{DATE}.png", dpi=fig.dpi)
    plt.suptitle(f"{experiment_type.title()}")
    plt.title("*2016 Adjusted compute estimates", fontsize=6, y=1.15)
    # plt.title("*Adjusted estimates")

def plot_histogram_observing_computing_ratio(usage_summary_dataframe, experiment_type):
    # category_counts = usage_summary_dataframe['cfg'].value_counts()
    # categories_to_remove = category_counts[category_counts < 6].index
    # Filter the DataFrame to keep only rows where the 'category' is NOT in the list of categories to remove
    # usage_summary_dataframe = usage_summary_dataframe[~usage_summary_dataframe['cfg'].isin(categories_to_remove)]
    fig = None
    # fig, gs, ax1, sc, sc_diff = plot_with_dataframe(usage=usage_summary_dataframe, use_task_data=False, use_edge_data=False,
    #                                        plot_type="hist",
    #                                        algorithms=['batch', 'heft'],
    #                                        labels={'labels': ['Batch', 'HEFT'], 'hatch': ['x', ''],
    #                                                    'color': ['silver', 'slateblue']}, rows=3, columns=1,
    #                                        gs_position=(0, 0))
    # ax1.set_xlabel("")
    # ax1.legend()
    # fig, gs2, ax2, sc, sc_diff = plot_with_dataframe(usage=usage_summary_dataframe, fig=fig, gs=gs, use_task_data=True,
    #                                         use_edge_data=False, plot_type="hist",
    #                                         algorithms=['batch', 'heft'],
    #                                         labels={'labels': ['Batch', 'HEFT'], 'hatch': ['x', ''],
    #                                                     'color': ['silver', 'slateblue']}, rows=3, columns=1,
    #                                         gs_position=(1, 0))
    # ax2.set_xlabel("")
    # # ax1.set_title("Without edge data")
    if not fig:
        gs2=None

    fig, gs, ax3, sc, sc_diff = plot_with_dataframe(usage=usage_summary_dataframe, fig=fig, gs=gs2, use_task_data=True,
                                           use_edge_data=True, plot_type="hist",
                                           algorithms=['batch', 'heft'],
                                           labels={'labels': ['Batch', 'HEFT'], 'hatch': ['x', ''],
                                                       'color': ['silver', 'slateblue']}, rows=1, columns=1)
                                           # gs_position=(2, 0))
    ax3.legend()
    # ax2.set_title("With edge data")
    ax3.set_xlabel("")

    handles, labels = [], []
    # for ax in [ax1, ax2, ax3]:
    for ax in [ax3]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    from collections import OrderedDict

    unique = OrderedDict(zip(labels, handles))

    # Establish limits based on maximum of two axes
    # ax1_lim = max(abs(y) for y in ax.get_ylim())
    # ax2_lim = max(abs(y) for y in ax2.get_ylim())
    ax3_lim = max(abs(y) for y in ax3.get_ylim())
    # lim = max(ax1_lim, ax2_lim, ax3_lim)
    lim = 30
    # from matplotlib.ticker import PercentFormatter
    # ax1.yaxis.set_major_formatter(PercentFormatter(1))
    # ax2.yaxis.set_major_formatter(PercentFormatter(1))
    # ax3.yaxis.set_major_formatter(PercentFormatter(1))
    # ax1.tick_params(labelbottom=False)
    # ax2.tick_params(labelbottom=False)
    # ax1.set_ylim([-lim, lim])
    # ax2.set_ylim([-lim, lim])
    ax3.set_ylim([-lim, lim])

    # ax1.set_xlimThis is often the best solution. Matplotlib will treat the main plot and the difference plot as a single group and reserve colorbar space for both together, avoiding overlap and keeping them aligned.([0, 4])
    # ax2.set_xlim([0, 4])
    # ax3.set_xlim([0, 4])

    fig.suptitle(f"{experiment_type.title()}")
    fig.supylabel("Num. Simulations", fontsize=8)
    fig.supxlabel("Computing time to observing time ratio", fontsize=8)
    fig.text(0.9, 0.75, "(a)")
    fig.text(0.9, 0.5, "(b)")
    fig.text(0.9, 0.25, "(c)")
    # for ax in [ax1, ax2, ax3]:
    for ax in [ax3]:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: abs(x))
        )
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # data1 = np.random.normal(1, 0.3, 1000)
    # data2 = np.random.normal(1.5, 0.3, 1000)
    #
    # bins = np.arange(0.0, 3, 0.2)
    #
    # plt.hist(data1, bins=bins, weights=np.ones(len(data1)), alpha=0.5)
    # plt.hist(data2, bins=bins, weights=-np.ones(len(data2)), alpha=0.5)
    #
    # plt.axhline(0, color='black')
    # plt.show()
    if SAVE_PLOTS:
        fig.savefig(f"Histogram-with-all-{experiment_type}-{DATE}.png", dpi=fig.dpi)
    # fig3.legend(unique.values(), unique.keys(), handleheight=3, handlelength=3,fontsize='small', bbox_to_anchor=(0.85, 0.88), framealpha=1.0, edgecolor='white')
    # fig.legend()

def plot_diff(usage, experiment_type):

    alg_list=['batch', 'heft']

    alg1, alg2 = alg_list[0], alg_list[1]
    xaxis = "plan_total_compute_flops"
    yaxis = "computing_to_observation_length_ratio"
    df1 = usage[usage["planning"] == alg1]
    df2 = usage[usage["planning"] == alg2]

    # sort by x for stability
    # Keep only the columns we need
    df0 = df1[[xaxis, yaxis]].copy()
    df1b = df2[[xaxis, yaxis]].copy()

    merged = (
        df0.merge(
            df1b,
            on=xaxis,
            suffixes=("_0", "_1")
        )
        .sort_values(xaxis)
    )

    merged["diff"] = merged[f"{yaxis}_0"] - merged[f"{yaxis}_1"]

    x_common = merged[xaxis].to_numpy()
    diff = merged["diff"].to_numpy()

    fig, gs = create_figure(1,1,twocolumn=False)
    ax_diff = fig.add_subplot()

    ax_diff.scatter(
        x_common,
        diff,
        color="black",
        # linewidth=2,
        label=f"{alg1} - {alg2}"
    )

    # ax_diff.axhline(-1, color="grey", linestyle="--", linewidth=1)
    ax_diff.set_ylabel("Diff-Y")

def plot_demand_vs_observation_ratio_scatter(usage_summary_dataframe, experiment_type):

    plot_index = 0
    handles = []
    labels = []
    handles_labels_collected = False  # Flag to only collect once
    usage_summary_dataframe['plan_total_compute_flops'] = usage_summary_dataframe['plan_total_compute']*1e15
    fig1, gs1, ax1, sc1, sc_diff1 = plot_with_dataframe(
        usage=usage_summary_dataframe, use_task_data=True, use_edge_data=True, plot_type="scatter",
        xaxis="plan_weighting", yaxis='computing_to_observation_length_ratio',
        algorithms={'batch': "Batch", 'heft': "HEFT"}, colors={'batch': 'red', 'heft': 'blue'},
        markers={"batch": 'v', "heft": 'o'}, fill=False,
    )
    fig2, gs2, ax2, sc2, sc_diff2 = plot_with_dataframe(
        usage=usage_summary_dataframe, use_task_data=True, use_edge_data=True, plot_type="scatter",
        xaxis="plan_total_compute_flops", yaxis='computing_to_observation_length_ratio',
        algorithms={'batch': "Batch", 'heft': "HEFT"}, colors={'batch': 'red', 'heft': 'blue'},
        markers={"batch": 'v', "heft": 'o'}, fill=False
    )
    fig3, gs3, ax3, sc3, sc_diff3 = plot_with_dataframe(
        usage=usage_summary_dataframe, use_task_data=True, use_edge_data=True, plot_type="scatter",
        xaxis="plan_total_data", yaxis='computing_to_observation_length_ratio',
        algorithms={'batch': "Batch", 'heft': "HEFT"}, colors={'batch': 'red', 'heft': 'blue'},
        markers={"batch": 'v', "heft": 'o'}, fill=False
    )


    # labels = ['Batch', "HEFT", f"(Batch − HEFT)"]
    ax1.set_ylim(0,2.6)
    ax2.set_ylim(0, 2.6)
    ax3.set_ylim(0, 2.6)
    ax1.set_xlim(0, 10)
    ax3.set_xlim(0, 1.8e11)
    fig1.suptitle(f"{experiment_type.title()}")
    fig2.suptitle(f"{experiment_type.title()}")
    fig3.suptitle(f"{experiment_type.title()}")# for i, a in enumerate(ax):
    #     a.set_ylim([0.0, 100.0])
    #     a.set_xlim([0.0, 100.0])
    #
    #     a.set_title(labels[i])
    #     plot_index += 1
    #     if not handles_labels_collected:
    #         handles, l = a.get_legend_handles_labels()
    #         handles_labels_collected = True
    cbar1 = fig1.colorbar(sc1, ax=ax1)
    cbar1.ax.set_ylabel("\% Large observations")
    # cbar1.set_label('% Large observations')
    cbar2 = fig2.colorbar(sc2, ax=ax2)
    cbar2.ax.set_ylabel("\% Large observations")
    cbar3 = fig2.colorbar(sc3, ax=ax3)
    cbar3.ax.set_ylabel("\% Large observations")
    # cbar2.set_label('% Large observations')
    #
    ax1.set_xlabel("Plan composition weighting")
    ax1.set_ylabel("Computing-to-observation weighting")
    ax2.set_xlabel("Total plan FLOPs")
    ax2.set_ylabel("Computing-to-observation weighting")
    ax3.set_xlabel("Total plan bytes")
    ax3.set_ylabel("Computing-to-observation weighting")
    import matplotlib.transforms as mtransforms
    for ax in [ax2, ax3]:
        offset = ax.xaxis.get_offset_text()
        offset.set_transform(
            offset.get_transform() +
            mtransforms.ScaledTranslation(
                15/72,  # no horizontal shift
                10 / 72.,  # 10 points upward
                fig2.dpi_scale_trans
            )
        )

    if SAVE_PLOTS:
        fig1.savefig(f"scatter-ratio-weight-{experiment_type}-{DATE}.png", dpi=fig1.dpi)
        fig2.savefig(f"scatter-ratio-compute-{experiment_type}-{DATE}.png", dpi=fig2.dpi)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def variance_decomposition_single_figure(
    df,
    x1="plan_total_compute",
    x2="plan_weighting",
    y="computing_to_observation_length_ratio",
    bin_width=0.25e6
):
    df = df.copy()
    df=df[df['planning'] == 'heft']

    # ----------------------------
    # 1. BIN X1 (fixed width)
    # ----------------------------
    min_edge = np.floor(df[x1].min() / bin_width) * bin_width
    max_edge = np.ceil(df[x1].max() / bin_width) * bin_width
    edges = np.arange(min_edge, max_edge + bin_width, bin_width)

    df["X1_bin"] = pd.cut(df[x1], bins=edges)
    df["X1_bin_str"] = df["X1_bin"].apply(lambda z: f"{z.left/1e6:.2f}-{z.right/1e6:.2f}M")

    # ----------------------------
    # 2. ANOVA (global structure)
    # ----------------------------
    groups = [g[y].values for _, g in df.groupby("X1_bin")]
    F, p = stats.f_oneway(*groups)

    # ----------------------------
    # 3. eta^2 (X1 effect size)
    # ----------------------------
    yvals = df[y].values
    grand_mean = np.mean(yvals)

    ss_between = 0.0
    ss_within = 0.0

    for _, g in df.groupby("X1_bin"):
        yi = g[y].values
        mi = yi.mean()
        ss_between += len(yi) * (mi - grand_mean) ** 2
        ss_within += np.sum((yi - mi) ** 2)

    eta_sq = ss_between / (ss_between + ss_within)

    # ----------------------------
    # 4. residualize Y by X1 (key step)
    # ----------------------------
    df["Y_residual"] = df[y] - df.groupby("X1_bin")[y].transform("mean")

    # ----------------------------
    # 5. within-bin X2 signal (R^2 proxy)
    # ----------------------------
    within_r2 = []
    for _, g in df.groupby("X1_bin"):
        if len(g) > 5:
            x2v = g[x2].values
            yv = g[y].values
            if np.std(x2v) > 0 and np.std(yv) > 0:
                r = np.corrcoef(x2v, yv)[0, 1]
                if not np.isnan(r):
                    within_r2.append(r**2)

    within_r2 = np.mean(within_r2) if len(within_r2) > 0 else np.nan

    # ----------------------------
    # 6. summary for trend line
    # ----------------------------
    summary = df.groupby("X1_bin_str")[y].mean().reset_index()

    # ----------------------------
    # 7. FIGURE
    # ----------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # ---- MAIN: boxplot (X1 → Y)
    df.boxplot(
        column=y,
        by="X1_bin_str",
        ax=ax,
        showfliers=False
    )

    ax.set_xlabel("plan_total_compute bins")
    ax.set_ylabel(y)
    ax.set_title("Structure in Y dominated by X1")

    # ---- trend line
    ax.plot(
        range(1, len(summary) + 1),
        summary[y].values,
        color="red",
        marker="o",
        linewidth=2,
        label="mean Y"
    )

    # ----------------------------
    # 8. INSET: X2 vs residual Y
    # ----------------------------
    ax_in = inset_axes(ax, width="40%", height="40%", loc="upper right")

    ax_in.scatter(
        df[x2],
        df["Y_residual"],
        alpha=0.2,
        s=10
    )

    ax_in.axhline(0, color="black", linestyle="--", linewidth=1)

    ax_in.set_title("Residual check", fontsize=9)
    ax_in.set_xlabel("X2", fontsize=8)
    ax_in.set_ylabel("Y residual", fontsize=8)

    # ----------------------------
    # 9. TITLE STATS
    # ----------------------------
    ax.text(
        0.02, 0.95,
        f"eta2(X1) = {eta_sq:.3f}\n"
        f"within-bin R2(X2) = {within_r2:.4f}\n"
        f"F = {F:.2f}, p = {p:.2e}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()
    plt.show()

    return {
        "F": F,
        "p": p,
        "eta_sq_X1": eta_sq,
        "within_bin_R2_X2": within_r2
    }

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def log_space_variance_analysis(
    df,
    x1="plan_total_compute",
    x2="plan_weighting",
    y="computing_to_observation_length_ratio"
):
    df = df.copy()
    df=df[df['planning'] == 'heft']
    # ----------------------------
    # 1. log transform X1
    # ----------------------------
    df["logX1"] = np.log10(df[x1])

    # ----------------------------
    # 2. fit smooth trend: Y ~ log(X1)
    # ----------------------------
    slope, intercept, r, p, _ = stats.linregress(df["logX1"], df[y])

    df["Y_hat_X1"] = intercept + slope * df["logX1"]

    # residual after removing X1 effect
    df["Y_residual"] = df[y] - df["Y_hat_X1"]

    # ----------------------------
    # 3. global R^2 (X1 effect)
    # ----------------------------
    r2_x1 = r**2

    # ----------------------------
    # 4. X2 vs residual relationship
    # ----------------------------
    r_x2_res = np.corrcoef(df[x2], df["Y_residual"])[0, 1]
    r2_x2_res = r_x2_res**2

    # ----------------------------
    # 5. significance test (X2 vs residual)
    # ----------------------------
    slope2, intercept2, r2, p2, _ = stats.linregress(df[x2], df["Y_residual"])

    # ----------------------------
    # 6. PLOT (single figure)
    # ----------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # ---- main: Y vs logX1 ----
    ax.scatter(df["logX1"], df[y], alpha=0.25, s=10, label="data")

    x_line = np.linspace(df["logX1"].min(), df["logX1"].max(), 200)
    y_line = intercept + slope * x_line

    ax.plot(x_line, y_line, color="red", linewidth=2, label="fit: Y ~ logX1")

    ax.set_xlabel("log10(plan_total_compute)")
    ax.set_ylabel(y)
    ax.set_title("Log-space scaling model + residual independence test")

    # ---- inset: X2 vs residual ----
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    ax_in = inset_axes(ax, width="40%", height="40%", loc="upper left")

    ax_in.scatter(df[x2], df["Y_residual"], alpha=0.2, s=10)
    ax_in.axhline(0, color="black", linestyle="--")

    ax_in.set_xlabel("X2", fontsize=8)
    ax_in.set_ylabel("Residual Y", fontsize=8)
    ax_in.set_title("X2 vs residual", fontsize=9)

    # ----------------------------
    # 7. annotation
    # ----------------------------
    ax.text(
        0.03, 0.97,
        f"R2(X1) = {r2_x1:.3f}\n"
        f"R2(X2 | X1) = {r2_x2_res:.4f}\n"
        f"p(X2) = {p2:.2e}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(facecolor="white", alpha=0.8)
    )

    plt.tight_layout()
    plt.show()

    return {
        "R2_logX1": r2_x1,
        "R2_X2_given_X1": r2_x2_res,
        "p_X2_given_X1": p2,
        "slope_logX1": slope
    }

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def trend_stability_plot_with_values(
        df,
        x1="plan_total_compute",
        x2="plan_weighting",
        y="computing_to_observation_length_ratio",
            n_bins=10
    ):

    df = df.copy()
    df=df[df['planning'] == 'heft']
    bin_width=0.25e6

    # ----------------------------
    # 1. BIN X1 (fixed width)
    # ----------------------------
    min_edge = np.floor(df[x1].min() / bin_width) * bin_width
    max_edge = np.ceil(df[x1].max() / bin_width) * bin_width
    edges = np.arange(min_edge, max_edge + bin_width, bin_width)

    df["X1_bin"] = pd.cut(df[x1], bins=edges)

    # df["X1_bin"] = pd.qcut(df[x1], q=n_bins, duplicates="drop")

    rows = []

    # sort bins in order
    bins = list(df["X1_bin"].cat.categories)

    prev_mean = None

    for i, b in enumerate(bins):

        sub = df[df["X1_bin"] == b]

        y_vals = sub[y]
        x2_vals = sub[x2]

        y_mean = y_vals.mean()
        y_std = y_vals.std()
        y_range = y_vals.max() - y_vals.min()

        x2_range = x2_vals.max() - x2_vals.min()

        # within-bin relative variation
        y_rel_var = y_std / (abs(y_mean) + 1e-12)

        # step in Y between X1 bins
        if prev_mean is None:
            delta_mean = np.nan
        else:
            delta_mean = y_mean - prev_mean

        rows.append({
            "X1_bin": str(b),
            "X2_range": x2_range,
            "Y_mean": y_mean,
            "Y_std": y_std,
            "Y_range": y_range,
            "Y_rel_variation_within_bin": y_rel_var,
            "Delta_Y_vs_previous_X1_bin": delta_mean,
            "n": len(sub)
        })

        prev_mean = y_mean

    df = pd.DataFrame(rows)
    df = df.copy()

    # avoid division issues
    ratio = df["Delta_Y_vs_previous_X1_bin"] / (df["Y_rel_variation_within_bin"] + 1e-12)

    plt.figure(figsize=(8,5))

    plt.plot(range(len(ratio)), ratio, marker="o")

    plt.yscale("log")  # critical: spans orders of magnitude

    plt.axhline(1, linestyle="--", color="red", label="Equal influence threshold")

    plt.xlabel("X1 bin index")
    plt.ylabel("X1 step change / X2 within-bin variation")
    plt.title("Dominance of X1 over X2 in explaining Y")

    plt.legend()
    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def within_bin_comparisons(df,
                           x1="plan_total_compute",
                           x2="plan_weighting",
                           y="computing_to_observation_length_ratio",
                           n_bins=10):

    df = df.copy()
    bin_width=0.25e6
    df=df[df['planning'] == 'batch']

    # ----------------------------
    # 1. BIN X1 (fixed width)
    # ----------------------------
    min_edge = np.floor(df[x1].min() / bin_width) * bin_width
    max_edge = np.ceil(df[x1].max() / bin_width) * bin_width
    edges = np.arange(min_edge, max_edge + bin_width, bin_width)

    df["X1_bin"] = pd.cut(df[x1], bins=edges)

    # df["X1_bin"] = pd.qcut(df[x1], q=n_bins, duplicates="drop")

    rows = []

    for b in df["X1_bin"].cat.categories:

        sub = df[df["X1_bin"] == b]

        x1_min = sub[x1].min()
        x1_max = sub[x1].max()
        x1_center = (x1_min + x1_max) / 2

        y_mean = sub[y].mean()
        y_std = sub[y].std()

        # within-bin X2 effect proxy = spread in Y
        y_within = sub[y].std()

        rows.append({
            "X1_center": x1_center,
            "Y_mean": y_mean,
            "Y_within_std": y_within,
            "n": len(sub)
        })

    table = pd.DataFrame(rows).sort_values("X1_center")
    print(table)
    # ----------------------------
    # plot in real units
    # ----------------------------
    fig, ax1 = plt.subplots(figsize=(8,6))

    ax1.plot(table["X1_center"], table["Y_mean"], marker="o", label="mean Y vs X1")
    ax1.set_xlabel("X1 (bin center)")
    ax1.set_ylabel("mean Y")

    # secondary axis = within-bin variation
    ax2 = ax1.twinx()
    ax2.scatter(table["X1_center"], table["Y_within_std"], color="red", marker="x", label="within-bin Y std")
    ax2.set_ylabel("within-bin Y variation (proxy for X2 + noise)")

    plt.title("X1-driven trend vs within-bin X2-induced variation")

    fig.tight_layout()
    plt.show()

    return table


SAVE_PLOTS = True

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    args = setup_parser()

    experiment_type = args.experiment_type
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
                                                       debug=False    # nonlinear_mediation_analysis(usage_summary_dataframe, experiment_type)
)
    # if df_total is None:

    df_total = pd.read_csv(df_total_path)
    if not simulation_summaries:
        with open("simulation_summaries.json") as fp:
            simulation_summaries = json.load(fp)

    if not df_summary.exists():
        usage_summary_dataframe = produce_summary_dataframe(df_total, base_dir)

        # Isolate shared configs to simulation results match
        batch_cfgs = set(usage_summary_dataframe[usage_summary_dataframe['planning'] == 'batch']['cfg'])
        heft_cfgs = set(usage_summary_dataframe[usage_summary_dataframe['planning'] == 'heft']['cfg'])
        common_cfgs = batch_cfgs & heft_cfgs
        usage_summary_dataframe = usage_summary_dataframe[usage_summary_dataframe['cfg'].isin(common_cfgs)]

        with df_summary.open("w") as fp:
            usage_summary_dataframe.to_csv(fp)
    else:
        usage_summary_dataframe = pd.read_csv(df_summary)

    usage_summary_dataframe = usage_summary_dataframe.drop_duplicates(subset=['cfg', 'planning'])
    ################################################################################
    ######                          MAKE PLOTS
    ################################################################################
    if experiment_type=="concurrent":
        groups = list(usage_summary_dataframe.groupby('cfg').groups.keys())
        # Select every second group name
        keep_groups = groups[::2]
        # Filter the DataFrame to keep only those groups
        usage_summary_dataframe = usage_summary_dataframe[usage_summary_dataframe['cfg'].isin(keep_groups)]

    LOGGER.info("Generating plots...")
    # plot_histogram_observing_computing_ratio(usage_summary_dataframe, experiment_type)
    # plot_demand_vs_observation_ratio_scatter(usage_summary_dataframe, experiment_type)
    # plot_flops_vs_demand_low(usage_summary_dataframe, 'heft', experiment_type)
    # perform_statistical_analysis(usage_summary_dataframe, experiment_type)
    # nonlinear_mediation_analysis(usage_summary_dataframe, experiment_type)
    # plot_diff(usage_summary_dataframe, experiment_type)
    print(within_bin_comparisons(usage_summary_dataframe))
    # TODO box plots across bins of petaflops categories
    #
    # observation_plans = get_observation_plans(df_total=df_total,
    #                                           config_dir=base_dir)

    # SHOW CONFIGS
    # plot_observation_plan(observation_plans.iloc[0], 'Test')

    # cfgs = select_n_configs_by_key(usage_summary_dataframe,  count=1
    # s = simulation_summaries[cfgs[0]]
    # create_simulation_schedule_map(pd.read_csv(StringIO(s)))

    # plan = observation_plans[(
    #         observation_plans["config"] == cfgs[0])]
    # plot_observation_plan(plan, 0.33)
    #
    # s = simulation_summaries[cfgs[0]]
    # create_simulation_schedule_map(pd.read_csv(StringIO(s)))

    plt.show()
