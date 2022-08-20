# Copyright (C) 11/5/22 RW Bunney

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

import json
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
from matplotlib.collections import PatchCollection
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Rectangle
from natsort import natsort_keygen

plt.rcParams.update(
    {"text.usetex": True, "font.family": "sans-serif", "font.size": 12,
     # "figure.autolayout": True,
     })

sns.set_palette("colorblind")

# name, max_compute
telescopes = [('low-adjusted', 896), ('mid-adjusted', 786)]

adjusted = False


# Calculate relative parametric time (i.e. if the parametric model was
# performed on N channels only.

def separate_data(results):
    """
    For each set of data, we want to keep a certain set of information,
    rather than repeat multiple bars.

    * For initial, we want three sets of information; parametric (either
    graph=base or garph=parallel,but we don't need both); workflow base,
    and workflow parallel.
    * For reduced scatter, we want workflow base and workflow parallel,
    and also parametric adjusted. Parametric adjusted is the equivalent time
    the parametric result would be if we ran on 512 machines. From this we see
    again that workflow=parallel provides us an approximate/equivalent time
    to the parametric model. The base graph (which has higher degrees of
    parallelism within each scatter) has a lower runtime; this is likely
    because it will 'steal' machines that are not in use with the additional
    paraellism.
    * For reduced nodes, we are confirming an assumption from the previous
    that the base model is machine stealing; this is demonstrated by
    restricting the nodes to the same as the scatter, leading to an equalised

    Hence the data we are working along is the following:

    * Initial; Nodes/Channels = 896; data = false, 
    parametric + workflow, graph= prototype + scatter 
    (parametric only has prototype, as it does not use the workflow (should
    be 'no graph')
    
    """

    hpsos = set(results['hpso'])
    yaxis = {}
    xaxis = {}

    return results


def create_standard_barplot_axis(ax, minor=True):
    """
    Create an axis and figure with standard grid structure and ticks

    Saves us having to reference/create an arbitrary

    Returns
    -------
    axis
    """

    # ax.yaxis.set_major_locator(ticker.MultipleLocator(10000))
    # # ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_major_formatter(FormatStrFormatter("%.0e"))
    # ax.grid(axis='both', which='major', color='grey')
    if minor:
        ax.grid(axis='both', which='minor', color='lightgrey',
                linestyle='dotted')
    ax.tick_params(right=True, top=True, which='both', direction='in')
    # ax.tick_params(axis='x', labelrotation=45.0)

    return ax


def plot_parametric_data(df, data=False, max_only=True):
    """
    There are two types of parametric data; 
        - Complete SDP: these are results associated with running on the
        maximum
        number of machines for that telescope
        - Adjusted: these are results calculated based on the number of machines
        that are expected. The parametric model is less flexible and we have
        written code on the fly here, so we want to use the time caculated
        and divide it by the fraction of the original/max that exists.

        E.g. for the 512 nodes, we expect:
            runtime(512)  =  runtime(896) / (512/896)

        for the parametric model.

    Parameters
    ----------
    df :
        Data frame

    Returns
    --------
    plot
        Matplotlib pyplot object
    """

    par_plot = None

    parametric_df = df[
        (df['simulation_type'] == 'parametric') & (df['data'] == data) & (
                df['graph'] == 'prototype')].sort_values(by='hpso',
                                                         key=natsort_keygen())

    workflow_scatter_df = df[
        (df['simulation_type'] == 'workflow') & (df['data'] == data) & (
                df['graph'] == 'scatter')].sort_values(by='hpso',
                                                       key=natsort_keygen())

    workflow_prototype_df = df[
        (df['simulation_type'] == 'workflow') & (df['data'] == data) & (
                df['graph'] == 'prototype')].sort_values(by='hpso',
                                                         key=natsort_keygen())

    dfs = {'par': parametric_df, 'scatter': workflow_scatter_df,
           'proto': workflow_prototype_df}

    # Maximum channels
    for d in dfs:
        max_df = dfs[d][(dfs[d]['nodes'] == 896) | (dfs[d]['nodes'] == 786)]

        # Maximum number of channels
        max_bool = ((max_df['nodes'] == 896) & (max_df['channels'] == 896) | (
                max_df['nodes'] == 786) & (max_df['channels'] == 786))
        dfs[d] = max_df[max_bool]

    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(9, 6))

    x = np.arange(0, 2 * len(dfs['par']['hpso']), step=2)
    axs[0][0] = create_standard_barplot_axis(axs[0][0], False)
    axs[0][0].bar(x, dfs['par']['time'], color="lightblue", zorder=1,
        edgecolor='black', width=0.8, log=True)
    axs[0][0].set_xticks(x, dfs['par']['hpso'])
    axs[0][0].set_ylabel(ylabel='', labelpad=10)
    axs[0][0].set_title('Parametric estimates\n (896 channels)')

    axs[0][1] = create_standard_barplot_axis(axs[0][1], False)
    axs[0][1].bar(x - 0.2, dfs['par']['time'], color="lightblue", zorder=1,
        edgecolor='black', width=0.4, log=True, alpha=0.5)
    axs[0][1].bar(x + 0.2, dfs['scatter']['time'], color='pink', zorder=1,
        edgecolor='black', width=0.4, log=True)
    axs[0][1].set_xticks(x, dfs['par']['hpso'])
    axs[0][1].set_ylabel(ylabel='', labelpad=10)
    axs[0][1].set_title('+ Scheduled scatter workflow \n (896 channels)')

    axs[0][2] = create_standard_barplot_axis(axs[0][2], False)
    axs[0][2].bar(x - 0.3, dfs['par']['time'], color="lightblue", zorder=1,
        edgecolor='black', width=0.3, log=True, alpha=0.5)
    axs[0][2].bar(x, dfs['scatter']['time'], color='pink', zorder=1,
        edgecolor='black', width=0.3, log=True, alpha=0.8)
    axs[0][2].bar(x + 0.3, dfs['proto']['time'], color='orange', zorder=1,
        edgecolor='black', width=0.3, log=True)
    axs[0][2].set_xticks(x, dfs['par']['hpso'])
    axs[0][2].set_ylabel(ylabel='', labelpad=10)
    axs[0][2].set_title('+ Scheduled prototype workflow \n (896 channels)')

    fig.supxlabel('HPSO')  # ,y=0.1)
    fig.supylabel('Runtime (s)')  # ,x=0.02)

    # 512 channels with parametric adjustement
    for d in dfs:
        if d == 'par': # we will use + adjust this, as 512 values don't
            # really exist
            continue
        max_df = dfs[d][(dfs[d]['nodes'] == 896) | (dfs[d]['nodes'] == 786)]

        # Maximum number of channels
        max_bool = ((max_df['nodes'] == 896) & (max_df['channels'] == 512) | (
                max_df['nodes'] == 786) & (max_df['channels'] == 512))
        dfs[d] = max_df[max_bool]

    x = np.arange(0, 2 * len(dfs['par']['hpso']), step=2)

    axs[1][0] = create_standard_barplot_axis(axs[1][0], False)
    axs[1][0].bar(x, dfs['par']['time'], color="lightblue", zorder=1,
        edgecolor='black', width=0.8, log=True)
    axs[1][0].set_xticks(x, dfs['par']['hpso'])
    axs[1][0].set_ylabel(ylabel='', labelpad=10)
    axs[1][0].set_title('Parametric estimates\n (896 channels)')

    (512 / 786)

    # 512 channels with 512 nodes, confirming the parallelism assumptions
    # about the prototype workflow generated by previous plots

    # fig.suptitle('Tmp title')
    # fig.marg
    # plt.margins(5)
    # ax2.bar(mid_x, mid_max['time'], color="lightblue", zorder=1,
    #         edgecolor='black', width=0.4,log=True)

    # ax2.set_xticks(mid_x, mid_max['hpso'])
    # ax1.set(ylim=(0,45000))
    # ax1.ticklabel_format(style='sci', scilimits=(4,4), axis='y')
    # ax2.set(ylim=(1,80000))
    # low_512channel_bool = (low_df['nodes'] == 896) & (low_df['channels'] ==
    # 512)
    # low_512channel = low_df[low_512channel_bool]
    # mid_512channel_bool = (mid_df['nodes'] == 786) & (mid_df['channels'] ==
    # 512)
    # mid_512channel = mid_df[mid_512channel_bool]
    """
    x = np.arange(len(low_512channel['hpso']))
    # Follow this approach moving forward
    # steps tomorrow - show initial run (above) in one row in a color (maybe
    # blue)
    # Next row is going to be the initial run (above, grey color) next to the
    # 'expected' runtime for the adjusted parametric model.
    # we can use this approach moving forward with some minor adjustments. 
    axs[2] = create_standard_barplot_axis(ax3, False)
    ax3.bar(x - 0.2, low_512channel['time'] * (512 / 786), color="pink",
            zorder=1, edgecolor='black', label='512 channels', width=0.4)  # ,
    # log=True)
    ax3.bar(x + 0.2, low_max['time'], color="lightgrey", zorder=1,
            edgecolor='black', width=0.4, label='Max channels')
    ax3.set_xticks(x, low_512channel['hpso'])
    # ax4.bar(mid_512channel['hpso'], mid_512channel['time'], color="lightgrey",
    #         zorder=1, edgecolor='black')  # , log=True)
    """
    # fig.tight_layout()
    # ax.bar()
    # ax1.xticks(rotation=45, ha="right")
    plt.show()

    return par_plot


def plot_workflow_data(df, graph="prototype", data=False):
    """
    We have 3 types of workflow data
    Parameters
    ----------
    data

    Returns
    -------

    """


def generate_plots():
    if adjusted:
        for t in telescopes:
            tel, max_comp = t
            results.loc[(results['model'] == 'parametric') & (
                    results['graph'] == 'parallel') & (
                                results['telescope'] == tel), 'time'] = results[
                                                                            (
                                                                                    results[
                                                                                        'model'] == 'parametric') & (
                                                                                    results[
                                                                                        'graph'] == 'parallel') & (
                                                                                    results[
                                                                                        'telescope'] == tel)][
                                                                            'time'] / (
                                                                                512 / max_comp)
        #
        # Updating the types because we are manipulating parametric to our
        # advantage
        # Parametric values are the same regardless of parallel/base,
        # so we can use
        # one set as an alternative selection of data
        results.loc[(results['model'] == 'parametric') & (
                results['graph'] == 'parallel'), ['graph',
                                                  'model']] = \
            'parametric_compute_adjusted'

    results.loc[results['model'] == 'parametric', 'graph'] = 'parametric'

    hatches = ['-', 'x', '\\']  # , '\\', '*', 'o']

    # Loop over the bars

    xaxis = set(results['hpso'])

    initial_parametric = initial[initial['model'] == 'parametric'][
        ['hpso', 'time']]

    g = sns.barplot(y='time', x='hpso', hue='graph', data=results)
    # g.set(yscale="log")
    for i, thisbar in enumerate(g.patches):
        # Set a different hatch for each bar
        thisbar.set_hatch(hatches[i % 3])


# plt.savefig(f'{str(curr_results).strip(curr_results.suffix)}_barplot.png')


if __name__ == '__main__':
    column_names = ['hpso', 'simulation_type', 'time', 'graph', 'telescope',
                    'nodes', 'channels', 'data']
    df_path = 'parametric_model_baselines/results_2022-08-11_locktest.csv'
    df = pd.read_csv(df_path, header=None, names=column_names)
    df = df.drop_duplicates()
    df['hpso'] = df['hpso'].str.strip('hpso0')
    plot = plot_parametric_data(df)
