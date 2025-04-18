#!/usr/bin/env python
# coding: utf-8

import h5py
import pandas
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path


def extract_simulations_from_hdf5(resfile, verbose=False):
    if not Path(resfile).exists():
        print("HDF5 file does not exist, leaving")
    store = pd.HDFStore(resfile)
    keysplit = []
    for k in store.keys():
        keysplit.append(k.split('/'))
    store.close()
    print(keysplit)
    dataset_types = ['sim', 'summary', 'tasks']
    simulations = {f"{e[1]}/{e[2]}": {d: None for d in dataset_types} for e in keysplit}
    for simulation, dtype in simulations.items():
        for dst in dataset_types:
            simulations[simulation][dst] = pd.read_hdf(resfile, key=f"{simulation}/{dst}")
    if verbose:
        for keys in simulations.keys():
            print(simulations.keys())
    return simulations


def collate_simulation_results(simulations):
    df_total = pd.DataFrame()
    for simulation, dtype in simulations.items():
        df = simulations[simulation]['summary']
        df_tel = (df[(df['actor'] == 'instrument')])
        obs_durations = []
        for obs in set(df_tel['observation']):
            df_obs = df_tel[df_tel['observation'] == obs]
            obs_durations.append(df_obs[df_obs['event'] == 'finished']['time'].iloc[0]
                                 - df_obs[df_obs['event'] == 'started']['time'].iloc[0])

        df_sim = simulations[simulation]['sim']

        # Get the simulation parameters from the configuration file.
        cfg_path = Path(df_sim['config'].iloc[0])
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

        # So long as the second last workflow is put on the scheduler
        # before the sum of the total observations is complete, we should be fine.
        # This means that the only thing that needs computing after the final observation
        # is the workflow associated with that observation, which means we aren't
        # 'in the red' as far as the shedule is concerned.

        # Get the second last stop time of a workflow on the scheduler
        df_sched = df[(df['actor'] == 'scheduler')]
        success = True
        if (sum(obs_durations)
            - sorted(df_sched[df_sched['event'] == 'stopped']['time'])[-2]) < 0:
            success = False

        parameters['success'] = success

        # Ratio of completion time to 'success criteria'.
        # Failed observations will report a negative number.
        parameters['success_ratio'] = (sum(obs_durations) - sorted(
            df_sched[df_sched['event'] == 'stopped']['time'])[-2]) / sum(obs_durations)

        # Use simulation config to differentiate between different sims
        parameters['sim_cfg'] = cfg_path.name
        df_total = pd.concat([df_total, parameters], ignore_index=True)

    return df_total


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
    df_total['demand'] = df_total['demand'].astype('str')
    # df_total['demand'] = sorted(df_total['demand'].astype('str'), key=int)
    return df_total


# fig, ax = plt.subplots()
# # ax.hist(ratios)

# scatter = ax.hist(demand['success'],color='blue') #, channels['success'], c='blue')
# scatter = ax.hist(demand['failure'], color='orange') #, channels['failure'], c='red')
# fig, ax = plt.subplots()

def produce_scatterplot(df_total):
    g = sns.scatterplot(x='demand', y='success_ratio', data=df_total)
    # # ax.set_xticks(ticks=range(0,512, 128), labels=range(0, 512,128))
    g = sns.displot(data=df_total, x='demand', hue='success',
                    col='channels')  # discrete=False) #, cbar=True,


def produce_distplot(df_total):
    g = sns.displot(data=df_total, x='demand', hue='failure',
                    multiple='dodge')  # , cbar=True,

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


import numpy as np


def create_simulation_schedule_map(simulations):
    df = simulations['Sun240616153636/skaworkflows_2024-06-16_15-06-51']['summary']
    df = simulations['Sun240616163628/skaworkflows_2024-06-16_14-41-49']['summary']
    df = simulations['Sun240616163628/skaworkflows_2024-06-16_15-35-55'][
        'summary']  # passes

    # 'Sun240616163628/skaworkflows_2024-06-16_14-41-49'
    actors = set(df['actor'])
    # Observation telescope, started/finished
    # observation buffer, start/end -> we don´t particularly care about buffer
    # observation scheduler, added/removed
    obs = set(df['observation'])
    inst, sched = {}, {}
    obs_d = {o: {} for o in obs}
    for o in obs_d:
        obs_d[o]['telescope'] = df[
            (df['observation'] == o) & (df['actor'] == 'instrument')]
        obs_d[o]['scheduler'] = df[
            (df['observation'] == o) & (df['actor'] == 'scheduler')]

    begin, end = [], []
    obs_list = []
    for o in sorted(obs):
        obs_list.append(f"{o}: 2: Scheduler")
        sdf = obs_d[o]['scheduler']
        begin.append(int(sdf[sdf['event'] == 'added']['time'].iloc[0]) * 5 / 3600)
        end.append(int(sdf[sdf['event'] == 'removed']['time'].iloc[0]) * 5 / 3600)
        obs_list.append(f"{o}: 1: Telescope")
        tdf = obs_d[o]['telescope']
        begin.append(int(tdf[tdf['event'] == 'started']['time'].iloc[0]) * 5 / 3600)
        end.append(int(tdf[tdf['event'] == 'finished']['time'].iloc[0]) * 5 / 3600)

    # import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.barh(range(len(begin)), np.array(end) - np.array(begin), color=['grey', 'orange'],
            left=np.array(begin), edgecolor='black')
    ax.set_yticks(range(len(begin)), obs_list)
    plt.show()


def get_observation_duration(df):
    df_tel = (df[(df['actor'] == 'instrument')])

    for obs in set(df_tel['observation']):
        df_obs = df_tel[df_tel['observation'] == obs]
        print(df_obs[df_obs['event'] == 'finished']['time'].iloc[0]
              - df_obs[df_obs['event'] == 'started']['time'].iloc[0])


def get_config_parameters(config):
    pass

if __name__ == '__main__':
    RESULT_FILE = 'results_f2024-07-23.h5'
    simulations = extract_simulations_from_hdf5(RESULT_FILE, verbose=True)
    if not simulations:
        exit(1)
    df_total = collate_simulation_results(simulations)
    df_total = convert_categorical_ints_to_str(df_total)

    produce_scatterplot(df_total)
    plt.show()
