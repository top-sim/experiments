#!/usr/bin/env python
# coding: utf-8

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

def extract_simulations_from_hdf5(resfile, verbose=True):
    path = Path(resfile)
    if not path.exists():
        print("HDF5 file does not exist, leaving")

    simulations = {}
    if path.is_dir():
        for p in path.iterdir():
            # TODO if this is a folder iterate through all hdf5 files
            tmp_simulations = {}
            store = pd.HDFStore(str(p))
            keysplit = []
            for k in store.keys():
                keysplit.append(k.split("/"))
            store.close()
            if verbose:
                print(p, keysplit)
            dataset_types = ["sim", "summary", "tasks"]
            tmp_simulations.update({
                f"{e[1]}/{e[2]}": {d: None for d in dataset_types} for e in keysplit
            })
            for simulation, dtype in tmp_simulations.items():
                for dst in dataset_types:
                    tmp_simulations[simulation][dst] = pd.read_hdf(
                        p, key=f"{simulation}/{dst}"
                    )
            simulations.update(tmp_simulations)
            if verbose:
                for keys in tmp_simulations.keys():
                    print(keys)
    return simulations


def collate_simulation_results(simulations):
    df_total = pd.DataFrame()
    for simulation, dtype in simulations.items():
        df = simulations[simulation]["summary"]
        df_tel = df[(df["actor"] == "instrument")]
        obs_durations = []
        # TODO does this only work for non-concurrent observations?
        for obs in set(df_tel["observation"]):
            df_obs = df_tel[df_tel["observation"] == obs]
            obs_durations.append(
                df_obs[df_obs["event"] == "finished"]["time"].iloc[0]
                - df_obs[df_obs["event"] == "started"]["time"].iloc[0]
            )

        df_sim = simulations[simulation]["sim"]

        # Get the simulation parameters from the configuration file.
        cfg_path = Path(df_sim["config"].iloc[0])

        # cfg_path = cfg_path.parent / str(cfg_path.parent / 'processed' / cfg_path.name)
        with open(cfg_path) as fp:
            cfg = json.load(fp)
        pipelines = cfg["instrument"]["telescope"]["pipelines"]
        nodes = len(cfg["cluster"]["system"]["resources"])
        parameters = (
            pd.DataFrame.from_dict(pipelines, orient="index")
            .reset_index()
            .rename(columns={"index": "observation"})
        )
        parameters["nodes"] = nodes  # Number of nodes

        wfs = list(parameters['workflow'])


        # So long as the second last workflow is put on the scheduler
        # before the sum of the total observations is complete, we should be fine.
        # This means that the only thing that needs computing after the final observation
        # is the workflow associated with that observation, which means we aren't
        # 'in the red' as far as the shedule is concerned.

        # Get the second last stop time of a workflow on the scheduler
        df_sched = df[(df["actor"] == "scheduler")]
        success = True
        second_last_index = -2
        if len(obs_durations) < 2:
            second_last_index = -1
        if (
                sum(obs_durations)
                - sorted(df_sched[df_sched["event"] == "stopped"]["time"])[
                    second_last_index
                ]
        ) < 0:
            success = False

        parameters["success"] = success

        # Ratio of completion time to 'success criteria'.
        # Failed observations will report a negative number.
        parameters["success_ratio"] = (
                                              sum(obs_durations)
                                              - sorted(
                                          df_sched[df_sched["event"] == "stopped"][
                                              "time"])[
                                                  second_last_index
                                              ]
                                      ) / sum(obs_durations)

        parameters["schedule_length"] = len(df_sim)
        parameters["planning"] = df_sim["planning"]
        parameters["scheduling"] = df_sim["scheduling"]

        # Use simulation config to differentiate between different sims
        parameters["sim_cfg"] = cfg_path.name
        parameters["total_obs_duration"] = sum(obs_durations)
        df_total = pd.concat([df_total, parameters], ignore_index=True)

    return df_total


def process_workflow_stats(config_dir: Path, workflow_list: list):
    """
    Go through each workflow config file related to the directory, and get summary
    information from them to describe the compute requirements for a given config.

    Parameters
    ----------
    config_dir

    Returns
    -------

    """



#     dfgb = df_total.groupby(['sim_cfg'])
#
# sims = []
# ratios = []
# for name, group in dfgb:
#     # group['success_ratio'].iloc[0])
#     sims.append(name[0])
#     # ratios.append(float(group['success_ratio'].iloc[0]))
#     # print(name, group[['success_ratio','demand', 'channels']])#.drop_duplicates())
#


# demand, channels = {'success': [], 'failure':[]}, {'success': [], 'failure':[]}
# for o in list(df_total['observation']):
#     if bool(df_total[df_total['observation'] == o]['success'].iloc[0]):
#         demand['success'].append(int(df_total[df_total['observation'] == o]['demand'].iloc[0]))
#         channels['success'].append(int(df_total[df_total['observation'] == o]['coarse_channels'].iloc[0]))
#     else:
#         demand['failure'].append(int(df_total[df_total['observation'] == o]['demand'].iloc[0]))
#         channels['failure'].append(int(df_total[df_total['observation'] == o]['coarse_channels'].iloc[0]))
# # df_total['coarse_channels'] = df_total['coarse_channels'].astype('category')


def convert_categorical_ints_to_str(df_total: pandas.DataFrame):
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


def pretty_print_simulation_results(simulations, key, verbose=True):
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


# fig, ax = plt.subplots()
# # ax.hist(ratios)

# scatter = ax.hist(demand['success'],color='blue') #, channels['success'], c='blue')
# scatter = ax.hist(demand['failure'], color='orange') #, channels['failure'], c='red')
# fig, ax = plt.subplots()


def produce_scatterplot(df_total):
    g = sns.scatterplot(x="demand", y="success_ratio", data=df_total)
    # # ax.set_xticks(ticks=range(0,512, 128), labels=range(0, 512,128))
    g = sns.displot(
        data=df_total, x="demand", hue="success", col="channels"
    )  # discrete=False) #, cbar=True,


def produce_distplot(df_total):
    g = sns.displot(
        data=df_total, x="demand", hue="failure", multiple="dodge"
    )  # , cbar=True,

    # bins=16) #, ax=ax) # , hue='success', ax=ax, dodge=True)
    # ax.set_xticks(range(-1, 12))
    # label_range = range(0, 576, 64)
    # ax.set_xticks(range(-1, len(label_range)-1))
    # ax.set_xticklabels(label_range)
    # # g.set_xticks(range(0, 512, 64))
    # g.set_xticklabels(range(0, 512, 64))
    # f, ax = plt.subplots()
    # dataset = df_total.pivot(index='demand', columns='channels', values='success_ratio')
    # sns.relplot(x='demand', y='coarse_channels', hue='success', data=df_total, col='nodes') # ,ax=ax)
    # print(df_total[['channels', 'coarse_channels', 'demand', 'success_ratio']])
    # sns


def create_simulation_schedule_map(simulations, key):
    df = simulations[key]["summary"]
    actors = set(df["actor"])
    # Observation telescope, started/finished
    # observation buffer, start/end -> we don´t particularly care about buffer
    # observation scheduler, added/removed
    obs = set(df["observation"])
    inst, sched = {}, {}
    obs_d = {o: {} for o in obs}
    for o in obs_d:
        obs_d[o]["telescope"] = df[
            (df["observation"] == o) & (df["actor"] == "instrument")
            ]
        obs_d[o]["scheduler"] = df[
            (df["observation"] == o) & (df["actor"] == "scheduler")
            ]

    # begin, end = [], []
    obs_list = [[], []]
    scheduler_start = []
    scheduler_end = []
    telescope_start = []
    telescope_end = []
    for o in sorted(obs):
        obs_list[0].append(f"{o}")  # Scheduler
        sdf = obs_d[o]["scheduler"]
        scheduler_start.append(
            int(sdf[sdf["event"] == "added"]["time"].iloc[0]) * 5 / 3600
        )
        scheduler_end.append(
            int(sdf[sdf["event"] == "removed"]["time"].iloc[0]) * 5 / 3600
        )

        obs_list[1].append(f"{o}")  # Telescope
        tdf = obs_d[o]["telescope"]
        telescope_start.append(
            int(tdf[tdf["event"] == "started"]["time"].iloc[0]) * 5 / 3600
        )
        telescope_end.append(
            int(tdf[tdf["event"] == "finished"]["time"].iloc[0]) * 5 / 3600
        )

    group_labels = ["Scheduler", "Telescope"]
    # import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    values = [[scheduler_start, scheduler_end], [telescope_start, telescope_end]]
    for i, (actor, group_label) in enumerate(zip(values, group_labels)):
        start, end = actor
        rects = ax.barh(
            range(i, len(start) * 2, 2),
            np.array(end) - np.array(start),
            label=group_label,
            left=np.array(start),
        )

        ax.set_yticks(range(i, len(start) * 2, 2), obs_list[i])
        # break
        # ax.barh(range(len(begin)), np.array(end) - np.array(begin),
        #         color=['grey', 'orange'],
        #         left=np.array(begin), edgecolor='black')
    ax.legend()


def get_observation_duration(df):
    df_tel = df[(df["actor"] == "instrument")]

    for obs in set(df_tel["observation"]):
        df_obs = df_tel[df_tel["observation"] == obs]
        print(
            df_obs[df_obs["event"] == "finished"]["time"].iloc[0]
            - df_obs[df_obs["event"] == "started"]["time"].iloc[0]
        )


def get_observation_plan_percentage(df_total, verbose=True):
    """
    Determine the resource usage of the telescope across the entire observation plan,
    as a fraction of the maximum possible value.

    Note
    -------
    This is to facilitate differentiating between various observation plan's use of the telescope.

    Currently this assumes LOW telescope
    """
    max_channels = 128 * 256
    max_demand = 512
    group = df_total.groupby(["sim_cfg", "planning"])
    usage = {}
    # Isolating observing plans, with planning algorithms
    for name, g in group:
        cfg, planning = name
        plan_demand = g["demand"].astype(int).sum()
        plan_channels = g["channels"].astype(int).sum()

        if max_demand in list(g["demand"]):
            print(max_demand)


        usage[f"{cfg}_{name}"] = {
            "demand_ratio": round(plan_demand / (max_demand * len(g)), 2),
            "channels_ratio": round(plan_channels / (max_channels * len(g)), 2),
            # "demand": g["demand"].iloc[0],
            # "channels": g["channels"].iloc[0],
            "data": g["data"].iloc[0],
            "success": g["success"].iloc[0],
            "schedule_length": int(g["schedule_length"].iloc[0]),
            "planning": planning,
            "schedule_length_ratio": (int(g["schedule_length"].iloc[0])
                                      / g['total_obs_duration'].iloc[0])
        }

    usage_df = pd.DataFrame(usage)
    if verbose:
        print(usage_df.T)
        # for name, results in usage.items():
        #     print(f"{name}: {results}")
    return usage_df.T


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

def plot_histogram_of_success(usage):

    """
    Plot the number of success for different bins based on demand ratio

    Bins = demand ratios

    To do this, we need to count the number of success/failures in each bins

    Parameters
    ----------
    usage

    Returns
    -------

    """
    data = []
    for demand in usage.groupby('demand_ratio'):
        group, df = demand
        data.extend([group] * (
            len(df[(df['success'] == True) & (df['planning'] != 'BatchPlanning')])))




def plot_scatter_comparison_plan_results(usage,
                                         verbose=True,
                                         withzoom=True,
                                         withannotations=True):
    """

    Create a scatterplot that shows relative completion time based on the simulation

    Parameters
    ----------
    usage
    verbose

    Returns
    -------

    """
    # sns.stripplot(
    #     data=usage, x="planning", y="schedule_length", hue="success"
    # )  # , style="planning")
    # sns.stripplot(data=usage, x="schedule_length_ratio", y="schedule_length", hue="success")

    from matplotlib.gridspec import GridSpec
    # Show only the successful results for batch_planning
    success=True
    planning = "BatchPlanning"
    # ax = plt.subplots(ncols=2, sharey=True)
    fig = plt.figure(layout="constrained")
    result = usage[(usage['success'] == success) & (usage['planning'] == planning)]
    gs = GridSpec(2,2, figure=fig,left=0.05, right=0.1, wspace=0.05)
    ax1=fig.add_subplot(gs[:,0])
    ax2=fig.add_subplot(gs[0,-1])
    ax3=fig.add_subplot(gs[-1,-1])
    ax1.scatter(
        np.array(result['demand_ratio']),
        np.array(result['schedule_length_ratio']),
        marker='o',
        label='Success + Batch', color='grey'
    )
    result_success_select = result[result['demand_ratio'] == 0.3].iloc[0]

    result = usage[(usage['success'] != success) & (usage['planning'] == planning)]
    ax1.scatter(
        np.array(result['demand_ratio']),
        np.array(result['schedule_length_ratio']),
        marker='+',
        label='Failure + Batch', color='red'
    )
    result_fail_select = result[result['demand_ratio'] == 0.3].iloc[-1]

    x1, x2, y1, y2 = 0.25, 0.35, 0.9, 2.0
    axins = ax1.inset_axes(
        [0.5, 0.5, 0.47, 0.47],
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[]
    )

    result = usage[(usage['success'] == success) & (usage['planning'] == planning)]
    axins.scatter(np.array(result['demand_ratio']),
                  np.array(result['schedule_length_ratio']), marker='o', color='grey')
    result = usage[(usage['success'] != success) & (usage['planning'] == planning)]
    axins.scatter(np.array(result['demand_ratio']),
                  np.array(result['schedule_length_ratio']), marker='+', color='red')
    import matplotlib.patheffects as path_effects
    from matplotlib.patches import ConnectionPatch

    con = ConnectionPatch(xyA=(0.3, result_success_select['schedule_length_ratio']+0.1),
                          xyB=(-0.05, 0.5), arrowstyle="->",
                          coordsA=axins.transData, coordsB=ax2.transData)
    fig.add_artist(con)
    # text = axins.annotate("Point test", xy=[0.3, result_success_select['schedule_length_ratio']], ha='center', size='small', xytext=(0.5,0.5),
    #     arrowprops = dict(
    #         arrowstyle="->", shrinkA=0, shrinkB=5, color="black", linewidth=0.75
    #     ),
    # )
    # text.set_path_effects(
    #     [path_effects.Stroke(linewidth=2, foreground="white"), path_effects.Normal()]
    # )
    # text.arrow_patch.set_path_effects(
    #     [path_effects.Stroke(linewidth=2, foreground="white"), path_effects.Normal()]
    # )
    # text = axins.annotate("Point test", xy=[0.3, result_fail_select['schedule_length_ratio']], ha='center', size='small', xytext=(0.5,1.5),
    #     arrowprops = dict(
    #         arrowstyle="->", shrinkA=0, shrinkB=5, color="black", linewidth=0.75
    #     ),
    # )
    # text.set_path_effects(
    #     [path_effects.Stroke(linewidth=2, foreground="white"), path_effects.Normal()]
    # )
    # text.arrow_patch.set_path_effects(
    #     [path_effects.Stroke(linewidth=2, foreground="white"), path_effects.Normal()]
    # )

    ax1.indicate_inset_zoom(axins, edgecolor='black')


    # HEFT STATIC SCHEDULING
    # result = usage[(usage['success'] == success) & (usage['planning'] != planning)]
    # ax[1].scatter(
    #     np.array(result['demand_ratio']),
    #     np.array(result['schedule_length_ratio']),
    #     marker='o',
    #     label='Success + Static', color='grey'
    # )
    #
    # result = usage[(usage['success'] != success) & (usage['planning'] != planning)]
    #


    # ax[1].scatter(
    #     np.array(result['demand_ratio']),
    #     np.array(result['schedule_length_ratio']),
    #     marker='+',
    #     label='Failure + Static', color='red'
    # )
    # for axn in ax:
    #     axn.yaxis.set_tick_params(labelleft=True)
    #     axn.set_xlim([0, 1.1])
    #     axn.legend()

    # Need to figure out how to match observation data points with the scheduling
    # algorithm data points; or just show the addition of the new points
    # For example

    # Histogram is going to be really important
    # It'll be the number of successful plans of a given variable in bins



    #
    # fig, ax = plt.subplots(ncols=2, sharey=True)
    # markers = ['o', 'X']
    # colors = ['red', 'black']
    # planning = set(usage['planning'])
    # data = set(usage['data'])
    # success = set(usage['success'])
    # # TODO invert the markers/colours so that colour is planning and marker is fail
    # for i, d in enumerate(data):
    #     lines = []
    #     for j, p in enumerate(planning):
    #         marker_lines = []
    #         for s in success:
    #             tmp_usage = usage[
    #                 (usage['planning'] == p)
    #                 & (usage['data'] == d)
    #                 & (usage['success'] == s)
    #                 ]
    #             marker = 'o' if s else 'X'
    #             line = ax[i].plot(
    #                 np.array(tmp_usage['channels_ratio']),
    #                 np.array(tmp_usage['schedule_length_ratio']),
    #                 marker=marker,
    #                 # facecolors='none',
    #                 # label=f"Planning: {p}",
    #                 color=colors[j]
    #             )
    #             marker_lines.append(line)
    #             if s:
    #                 lines.append(line)
    #
    #             ax[i].legend()
    #     #     ax[1].add_artist(
    #     #         ax[1].legend(
    #     #             handles=marker_lines,
    #     #             labels=success,
    #     #             loc='center right',
    #     #             bbox_to_anchor=(1.5, 0.5),
    #     #             title='Success')
    #     #     )
    #     #     # lines.append(line)
    #     # ax[1].legend(
    #     #     handles=lines, labels=planning, title='Allocation policy',
    #     #     bbox_to_anchor=(1.5, 1.0),
    #     # )  # =lines)
    #     # ax[1].set_title(f"Data = {d}")


def plot_comp_per_channels(df_total):
    """
    Demonstrate how comp per channels goes down which explains why failure happens
    more in the lower channel bins.
    Parameters
    ----------
    usage

    Returns
    -------

    """

    planning = set(usage['planning'])
    success = set(usage['success'])

    usage = usage[usage['data'] == False]





def calculate_maximum_moving_average_for_observing_plan():
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



def plot_observation_plan(usage):
    pass


def get_config_parameters(config):
    pass


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    RESULT_FILE = sys.argv[
        1
    ]  # "/home/rwb/Dropbox/University/PhD/experiment_data/chapter4/scalability/low/results/results_f2024-08-03.h5"

    re_load = bool(int(sys.argv[2]))
    df_total_path = Path('df_total.csv')
    df_total = None
    if not df_total_path.exists() or re_load:
        # Trui RESULT_FILE = 'results_f2024-07-23.h5'
        simulations = extract_simulations_from_hdf5(RESULT_FILE, verbose=True)
        if not simulations:
            exit(1)
        df_total = collate_simulation_results(simulations)
        df_total = convert_categorical_ints_to_str(df_total)
        print(df_total)
        with open('df_total.csv', 'w') as fp:
            df_total.to_csv(fp)
    # for s in siulations:
    #     create_simulation_schedule_map(simulations, s)
    else:
       df_total = pd.read_csv(df_total_path)
    usage_summary_dataframe=get_observation_plan_percentage(df_total)
    with open('usage_summary.csv', 'w') as fp:
        usage_summary_dataframe.to_csv(fp)
    plot_scatter_comparison_plan_results(usage_summary_dataframe)
    # produce_scatterplot(df_total)

    plt.show()
