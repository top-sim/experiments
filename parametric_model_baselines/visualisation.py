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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.collections import PatchCollection
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16,
    # "figure.autolayout": True,
})

results_896 = pd.read_csv('parametric_model_baselines/2022_05_10_output.csv')
results_512 = pd.read_csv(
    'parametric_model_baselines/2022_05_11_512channels_output.csv'
)

# Drop the method=workflow, graph=parallel from results_512 and replace with
# results_896 data
# results_512 = results_512.drop(
#     results_512[
#         (results_512['model'] == 'workflow')
#         & (results_512['graph'] == 'parallel')
#         ].index
# )
# results_896_subset = results_896[
#     (results_896['model'] == 'workflow')
#     & (results_896['graph'] == 'parallel')
# ]
results_512.loc[
    (results_512['model'] == 'parametric')
    & (results_512['graph'] == 'parallel'), 'time'
] = results_512[
        (results_512['model'] == 'parametric')
        & (results_512['graph'] == 'parallel')
    ]['time'] / (512 / 896)

# Updating the types because we are manipulating parametric to our advantage
# Parametric values are the same regardless of parallel/base, so we can use
# one set as an alternative selection of data
# TODO Update this to be an extra column that we control based on a
#  combination of model/graph
results_512.loc[
    (results_512['model'] == 'parametric')
    & (results_512['graph'] == 'parallel'), 'graph'
] = 'parametric_compute_adjusted'
results_512.loc[
    (results_512['model'] == 'parametric')
    & (results_512['graph'] == 'parametric_compute_adjusted'), 'model'
] = 'parametric_compute_adjusted'

results_512.loc[
    results_512['model'] == 'parametric', 'graph'
] = 'parametric'
g = sns.catplot(y='time', x='hpso', hue='graph', kind='bar', data=results_512)
g.set(yscale="log")
plt.show()

# with open("parametric_model_baselines/low_base/workflows_896channels"
#           "/hpso01_time-18000_channels-896_tel-512_no_data.json") as f:
#     jdict = json.load(f)

# ngraph = nx.readwrite.json_graph.node_link_graph(jdict['graph'])
# task_dict = {}
# for n in ngraph.nodes:
#     name = n.split("_")[1]
#     if name in task_dict:
#         continue
#     else:
#         task_dict[name] = ngraph.nodes[n]['comp']
#
# node_compute = 10726000000000.0
#
# ngraph2 = nx.readwrite.json_graph.node_link_graph(jdict['graph'])
# task_dict2 = {}
# for n in ngraph2.nodes:
#     name = n.split("_")[1]
#     if name in task_dict2:
#         continue
#     else:
#         task_dict2[name] = ngraph2.nodes[n]['comp']
#
# task_dict['channels'] = 896
# task_dict2['channels'] = 512
# task_dict['nodes']=len(ngraph)
# task_dict2['nodes']=len(ngraph2)
# data=([[n,task_dict[n],task_dict['channels'],task_dict['nodes']] for n in task_dict])
# data + [[n,task_dict2[n],task_dict2['channels'],task_dict2['nodes']] for n in task_dict2]
# {data[n].append(int(task_dict[n])) for n in task_dict}
# {data[n].append(int(task_dict2[n])) for n in task_dict}
# g = sns.catplot(x='task', y='time/node(s)', hue='graph', data=df)
# grouped_df = results.groupby(['planning', 'delay', 'config']).size().astype(
#         float
#     ).reset_index(name='time').sort_values(by=['planning'])
#     grouped_df['config'] = grouped_df['config'].str.replace(
#         '2021_isc-hpc/config/single_size/40cluster/mos_sw', '').str.strip(
#         '.json').astype(float)
