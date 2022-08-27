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
    -------)
    axis
    """

    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if minor:
        ax.grid(axis='both', which='minor', color='lightgrey',
                linestyle='dotted')
    ax.tick_params(right=True, top=True, which='both', direction='in')
    return ax


def plot_data(df, data=False, logarithmic=True, max_only=True):
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

    parametric_df = df[
        (df['simulation_type'] == 'parametric') & (df['data'] == data) & (
                df['graph'] == 'prototype')].sort_values(by='hpso',
                                                         key=natsort_keygen())

    parametric_df['time'] = parametric_df['time'] # /1000
    workflow_scatter_df = df[
        (df['simulation_type'] == 'workflow') & (df['data'] == data) & (
                df['graph'] == 'scatter')].sort_values(by='hpso',
                                                       key=natsort_keygen())
    workflow_scatter_df['time'] = workflow_scatter_df['time'] #  / 1000
    workflow_prototype_df = df[
        (df['simulation_type'] == 'workflow') & (df['data'] == data) & (
                df['graph'] == 'prototype')].sort_values(by='hpso',
                                                         key=natsort_keygen())
    workflow_prototype_df['time'] = workflow_prototype_df['time']#  / 1000

    dfs = {'par': parametric_df, 'scatter': workflow_scatter_df,
           'proto': workflow_prototype_df}

    # Maximum channels
    for d in dfs:
        max_df = dfs[d][(dfs[d]['nodes'] == 896) | (dfs[d]['nodes'] == 786)]

        # Maximum number of channels
        max_bool = ((max_df['nodes'] == 896) & (max_df['channels'] == 896) | (
                max_df['nodes'] == 786) & (max_df['channels'] == 786))
        dfs[d] = max_df[max_bool]

    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(16, 10), gridspec_kw={
        "hspace":0.6},layout="tight")

    fig.supxlabel('HPSO' , y=0.025)
    fig.supylabel('Scheduled makespan (s)' , x=0.05)

    x = np.arange(0, 2 * len(dfs['par']['hpso']), step=2)
    axs[0][0] = create_standard_barplot_axis(axs[0][0], False)
    axs[0][0].bar(x, dfs['par']['time'] , color="lightblue", zorder=1,
                  edgecolor='black', width=1.8, log=logarithmic, label='Parametric')
    axs[0][0].set_xticks(x, dfs['par']['hpso'])
    axs[0][0].set_title('Parametric estimates\n (896 channels)')
    axs[0][0].set_xlabel(xlabel='(a)')

    # Scatter
    axs[0][1] = create_standard_barplot_axis(axs[0][1], False)
    axs[0][1].bar(x - 0.45, dfs['par']['time'] , color="lightblue",
                  zorder=1, edgecolor='black', width=0.9, log=logarithmic, alpha=0.5, label='Parametric')
    axs[0][1].bar(x + 0.45, dfs['scatter']['time'] , color='pink', zorder=1,
                  edgecolor='black', width=0.9, log=logarithmic, label='Workflow, Scatter')
    axs[0][1].set_xticks(x, dfs['par']['hpso'])
    # axs[0][1].set_ylabel(ylabel='', labelpad=10)
    axs[0][1].set_xlabel(xlabel='(b)')
    axs[0][1].set_title('+ Scheduled scatter workflow \n (896 channels)')

    # Prototype
    axs[0][2] = create_standard_barplot_axis(axs[0][2], False)
    axs[0][2].bar(x - 0.6, dfs['par']['time'] , color="lightblue", zorder=1,
                  edgecolor='black', width=0.6, log=logarithmic, alpha=0.5, label="Parametric")
    axs[0][2].bar(x, dfs['scatter']['time'] , color='pink', zorder=1,
                  edgecolor='black', width=0.6, log=logarithmic, alpha=0.8, label="Workflow, Scatter")
    axs[0][2].bar(x + 0.6, dfs['proto']['time'] , color='orange', zorder=1,
                  edgecolor='black', width=0.6, log=logarithmic, label="Workflow, Protoype")
    axs[0][2].set_xticks(x, dfs['par']['hpso'])
    axs[0][2].set_xlabel(xlabel='(c)')
    axs[0][2].set_title('+ Scheduled prototype workflow \n (896 channels)')

    # Generate bar legend
    handles, labels = axs[0][2].get_legend_handles_labels()
    lgd = axs[0][2].legend(handles, labels, loc='upper center',bbox_to_anchor=(-0.7, -0.15))

    # 512 channels with parametric adjustement
    dfs = {'par': parametric_df, 'scatter': workflow_scatter_df,
           'proto': workflow_prototype_df}

    for d in dfs:
        max_df = dfs[d][(dfs[d]['nodes'] == 896) | (dfs[d]['nodes'] == 786)]
        # if d == 'par': # we will use + adjust this, as 512 values don't
        #     dfs[d] = max_df
        #     continue

        # Maximum number of channels
        max_bool = ((max_df['nodes'] == 896) & (max_df['channels'] == 512) | (
                max_df['nodes'] == 786) & (max_df['channels'] == 512))
        dfs[d] = max_df[max_bool]

    x = np.arange(0, 2 * len(dfs['par']['hpso']), step=2)

    axs[1][0] = create_standard_barplot_axis(axs[1][0], False)
    axs[1][0].bar(x - 0.6, dfs['par']['time'] , color="lightgrey", zorder=1,
                  edgecolor='black', width=0.6, alpha=0.8, log=logarithmic,label='Non-adjusted Parametric')
    axs[1][0].set_xticks(x, dfs['par']['hpso'])

    dfs['par'].loc[df['telescope'] == 'low-adjusted', 'time'] = (
            dfs['par']['time']/(512/896)
    )
    dfs['par'].loc[df['telescope'] == 'mid-adjusted', 'time'] = (
            dfs['par']['time'] / (512 / 786))
    axs[1][0].bar(x, dfs['par']['time'] ,
                  color="lightblue", zorder=1, edgecolor='black', width=0.6,
                  log=logarithmic)
    axs[1][0].bar(x + 0.6, dfs['scatter']['time'] , color='pink', zorder=1,
                  edgecolor='black', width=0.6, log=logarithmic)
    axs[1][0].set_xticks(x, dfs['par']['hpso'])
    axs[1][0].set_xlabel(xlabel='(c)')
    axs[1][0].legend(loc='upper left')
    axs[1][0].set_title('Adjusted parametric estimates\n '
                        '+ Scheduled scatter workflow\n '
                        '(512 channels, 896 nodes)')

    # Second row, where we introduce the variable scatters
    axs[1][1] = create_standard_barplot_axis(axs[1][1], False)
    axs[1][1].bar(x-0.4, dfs['par']['time'], color="lightblue",
                  zorder=1, edgecolor='black', width=0.4, log=logarithmic)
    axs[1][1].bar(x, dfs['scatter']['time'], color='pink', zorder=1,
                  edgecolor='black', width=0.4, log=logarithmic)
    axs[1][1].bar(x + 0.4, dfs['proto']['time'], color='orange', zorder=1,
                  edgecolor='black', width=0.4, log=logarithmic)

    axs[1][1].set_xticks(x, dfs['par']['hpso'])
    axs[1][1].set_xlabel(xlabel='(d)')
    # axs[1][1].set_ylabel(ylabel='', labelpad=10)
    axs[1][1].set_title('+ Scheduled prototype estimates\n '
                        '(512 channels, 896 nodes)')

    # Calculate percentage speed improvement of more parallel prototype workflow.
    # This produces 0.6-3.1% speed improvement
    #[0.03086936 0.02958504 0.02958504 0.01541825 0.01940492 0.00592531 0.01103956]
    print(1.0-np.array(dfs['proto']['time'])/ np.array(dfs['scatter']['time']))

    # 512 channels with 512 nodes, confirming the parallelism assumptions
    # about the prototype workflow generated by previous plots
    # Demonstrate prototype 512 node execution as equiv. to scatter on 896
    dfs = {'par': parametric_df, 'scatter': workflow_scatter_df, 'proto':workflow_prototype_df}

    for d in dfs:
        max_df = dfs[d][(dfs[d]['nodes'] == 512) | (dfs[d]['nodes'] == 512)]

        max_bool = ((max_df['nodes'] == 512) & (max_df['channels'] == 512) | (
                max_df['nodes'] == 512) & (max_df['channels'] == 512))
        dfs[d] = max_df[max_bool]

    axs[1][2] = create_standard_barplot_axis(axs[1][2], False)
    dfs['par'].loc[df['telescope'] == 'low-adjusted', 'time'] = (
            dfs['par']['time']/(512/896)
    )
    dfs['par'].loc[df['telescope'] == 'mid-adjusted', 'time'] = (
            dfs['par']['time'] / (512 / 786))

    axs[1][2].bar(x-0.4, dfs['par']['time'] , color="lightblue", zorder=1,
                  edgecolor='black', width=0.4, log=logarithmic)
    axs[1][2].bar(x, dfs['scatter']['time'] , color='pink', zorder=1,
                  edgecolor='black', width=0.4, log=logarithmic)
    axs[1][2].bar(x + 0.4, dfs['proto']['time'] , color='orange', zorder=1,
                  edgecolor='black', width=0.4, log=logarithmic)

    axs[1][2].set_xticks(x, dfs['par']['hpso'])
    axs[1][2].set_title('Scheduled prototype estimates,\n'
                        'Reduced nodes\n'
                        ' (512 channels, 512 nodes)')
    plt.savefig("chapter5/parametric_model_baselines/plot_parametric_baseline_comparisons_nodata.eps")
    plt.savefig("chapter5/parametric_model_baselines/plot_parametric_baseline_comparisons_nodata.png",dpi=150)


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
    df_path = 'chapter5/parametric_model_baselines/results_2022-08-11_locktest.csv'
    df = pd.read_csv(df_path, header=None, names=column_names)
    df = df.drop_duplicates()
    df['hpso'] = df['hpso'].str.strip('hpso0')
    plot = plot_data(df)
