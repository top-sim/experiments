# Copyright (C) 31/8/21 RW Bunney

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import sys
import json
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Rectangle

import networkx as nx

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16,
    "figure.autolayout": True,
})

LOGGER = logging.getLogger(__name__)


def create_standard_axis(ax, minor=True):
    """
    Create an axis and figure with standard grid structure and ticks

    Saves us having to reference/create an arbitrary

    Returns
    -------
    axis
    """

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(axis='both', which='major', color='grey')
    if minor:
        ax.grid(axis='both', which='minor', color='lightgrey',
                linestyle='dotted')
    # ax.tick_params(
    #     right=True, top=True, which='both', direction='in'
    # )
    #
    return ax


def plot_task_runtime(
        df, config, alt_axes=None, **kwargs):
    """
    For the provided dataframe and config file, plot the number of running
    tasks at any given time

    Parameters
    ----------
    df
    config
    alt_axes :
        Alternative axis exists, then we plot onto that axis instead.
        Helps us create multiple plots
    color
    alpha

    Returns
    -------
    plot : The plot
    """

    # select the specific workflow based on the config entry
    if alt_axes:
        fig, ax1, ax2 = alt_axes
        config = config.lstrip('publications/')
    else:
        fig, (ax1, ax2) = plt.subplots(figsize=(15, 8),
                                       nrows=2, ncols=1, sharex=True,
                                       gridspec_kw={
                                           'hspace': 0.1,
                                           'height_ratios': [3, 1]}
                                       )
        ax2.set_xlabel('Simulation Time (min)')
        ax1.set_ylabel('No. Running Tasks')
        ax1 = create_standard_axis(ax1)
        ax2 = create_standard_axis(ax2, minor=False)
        ax2.set_ylabel('Observation workflow')

    if kwargs:
        color = kwargs['color']
        alpha = kwargs['alpha']
    else:
        color = 'red'
        alpha = 1.0
    wf_df = df[df['config'] == config]
    x = np.array(wf_df.index)
    y = np.array(wf_df['running_tasks'])
    ax1.plot(x, y, color=color, alpha=alpha)
    ax1.set_ylim(ymin=0)
    ax1.set_xlim(xmin=0)  # , xmax=550)
    return fig, ax1, ax2


def find_workflow_boundaries(tasks_sim, wf_config):
    """
    Using the tasks dataframe, find the boundaries of when the observation
    workflow started and finished
    Parameters
    ----------
    tasks_df

    Returns
    -------

    """

    # tasks_df = pd.read_pickle(
    #     'simulation_output/2021-08-09_global_tasks_batch.pkl')
    tasks_df = pd.read_pickle(
        tasks_sim
    )
    tasks_df = tasks_df.reset_index(level=0)
    tasks_df = tasks_df.rename(columns={'index': 'tasks'})
    # remove ingest tasks from consideration
    observations = set(tasks_df['observation_id'])

    wf_boundaries = {}
    for o in observations:
        focus = tasks_df[tasks_df['observation_id'] == o]

        focus = focus[focus['config'].str.contains(wf_config)]

        focus = focus[~focus['tasks'].str.contains('ingest')]
        ast_wf = focus.sort_values(by='ast').iloc[0]['aft']
        aft_wf = focus.sort_values(by='aft', ascending=False).iloc[0]['aft']
        wf_boundaries[o] = (ast_wf, aft_wf)

    return wf_boundaries


def generate_groupby_dataframes(df, grouby_list=None):
    """
    Given a list of columns to groupby, produce a dataframe that groups the
    columns.

    This is useful for generating total_simtime values
    Parameters
    ----------
    df : pd.DataFrame
        Original Simulation dataframe

    grouby_list : list
        List of column attributes that you want to group by

    Returns
    -------

    """
    grouped_df = df.groupby(['planning', 'delay', 'config']).size().astype(
        float
    ).reset_index(name='time').sort_values(by=['planning'])
    grouped_df['config'] = grouped_df['config'].str.replace(
        '2021_isc-hpc/config/single_size/40cluster/mos_sw', '').str.strip(
        '.json').astype(float)


def translate_existing_dataframes(df):
    """
    Older data frames have incorrect object types; we can update them to
    improve their usability

    Parameters
    ----------
    df

    Returns
    -------

    """


if __name__ == '__main__':
    os.chdir("/home/rwb/github/thesis_experiments")
    DATA_DIR = 'visualisation_playground/playground_data/'

    # Keywords for differentiating simulation output and tasks output
    SIM_KEYWORD = 'sim'
    TASKS_KEYWORD = 'tasks'
    sim = {
        'error': {
            'sim': f'{DATA_DIR}/2021-08-09_global_sim_batch.pkl',
            'tasks': f'{DATA_DIR}/2021-08-09_global_tasks_batch.pkl',
            'config': 'mos_sw80.json'
        },
        'updated': {
            'sim': f"{DATA_DIR}/2021-09-07-sim.pkl",
            'tasks': f"{DATA_DIR}/2021-09-07-tasks.pkl",
            'config': f'mos_sw80.json'
        }
    }

    # TODO check for existence of configuration file to ensure the simulation
    #  works.
    for version in sim:
        with open(
                "archived_results/2021_isc-hpc/config/workflows/shadow_Continuum_ChannelSplit_80.json",
                'r') as jfile:
            wf = json.load(jfile)
            graph = nx.readwrite.json_graph.node_link_graph(wf['graph'])

        os.listdir(DATA_DIR)
        ex_file = '2021-08-09_global_sim_batch.pkl'
        # ex_file = "batch_allocation_experiments/2021-09-07-sim.pkl"
        batch_df = pd.read_pickle(sim[version]['sim'])
        objects = [
            'observation_queue', 'schedule_status', 'delay_offset',
            'algtime', 'planning', 'scheduling', 'config'
        ]

        if len(batch_df['config'].str.contains(sim[version]['config'])) == 0:
            LOGGER.info(f"Skipping {sim[version]['sim']} as "
                        f"{sim[version]['config']} is not present in "
                        f"simulation")
        ex_wf_config = sim[version]['config']
        wf_bd = find_workflow_boundaries(sim[version]['tasks'], ex_wf_config)

        fig, ax1, ax2 = plot_task_runtime(batch_df, ex_wf_config)

        realloc_file = '2021-07-08_global_sim_greedy_from_plan.pkl'
        realloc_df = pd.read_pickle(f'{DATA_DIR}/{realloc_file}')
        realloc_df_fcfs = realloc_df[realloc_df['planning'] == 'fcfs']
        realloc_wf_wd = find_workflow_boundaries(
            f'{DATA_DIR}/2021-07-08_global_tasks_greedy_from_plan.pkl',
            ex_wf_config.lstrip('archived_results/')
        )
        # plt.show()
        fig, ax1, ax2 = plot_task_runtime(
            realloc_df_fcfs, ex_wf_config, (fig, ax1, ax2), color='blue',
            alpha=0.4)
        ax1.legend(['Batch', 'FCFS w/ Realloc'])

        ast = wf_bd['wallaby'][0]
        aft = wf_bd['wallaby'][1]
        data_dict = {'obs': [], 'width': [], 'left': []}
        for obs in wf_bd:
            ast, aft = wf_bd[obs]
            data_dict['obs'].append(obs)
            data_dict['width'].append(aft - ast)
            # the x-coordinates of left-hand side of bar, see axes.barh
            data_dict['left'].append(ast)

        y = np.arange(len(data_dict['obs']))
        ax2.barh(y - 0.25, data_dict['width'],
                 left=data_dict['left'],
                 color=['teal'], height=0.5, label='batch')

        realloc_data_dict = {'obs': [], 'width': [], 'left': []}
        for obs in data_dict['obs']:
            ast, aft = realloc_wf_wd[obs]
            realloc_data_dict['obs'].append(obs)
            realloc_data_dict['width'].append(aft - ast)
            # the x-coordinates of left-hand side of bar, see axes.barh
            realloc_data_dict['left'].append(ast)

        ax2.barh(y + 0.25, realloc_data_dict['width'],
                 left=realloc_data_dict['left'],
                 color=['orange'], height=0.5, label='realloc')

        labels = data_dict['obs']
        ax2.set_axisbelow(True)
        ax2.set_yticks(y)
        ax2.set_yticklabels(labels)

        # ax2.set(yticklabels=labels)

        ax2.legend()
        fig.align_ylabels()
        plt.figure(figsize=(16, 8))
        plt.show()
