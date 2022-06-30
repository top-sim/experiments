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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.collections import PatchCollection
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Rectangle

plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.size": 16,
     # "figure.autolayout": True,
     })

sns.set_palette("colorblind")

def collate_results():


    initial = pd.read_csv('parametric_model_baselines/2022_05_10_output.csv')
    initial['experiment'] = 'initial'
    reduced_scatter = pd.read_csv(
        'parametric_model_baselines/2022_05_11_512channels_output''.csv')
    reduced_scatter['experiment'] = 'reduced_scatter'
    reduced_nodes = pd.read_csv(
        'parametric_model_baselines/2022_05_31_output_512nodes.csv')
    reduced_nodes['experiment'] = 'reduced_nodes'

    results = pd.DataFrame()

# name, max_compute
telescopes = [('low-adjusted', 896), ('mid-adjusted', 786)]

adjusted = False
# Calculate relative parametric time (i.e. if the parametric model was
# performed on N channels only.

def separate_data():
    """
    For each set of data, we want to keep specific information


    """
    return data


if adjusted:
    for t in telescopes:
        tel, max_comp = t
        results.loc[
            (results['model'] == 'parametric')
            & (results['graph'] == 'parallel')
            & (results['telescope'] == tel), 'time'
        ] = results[
            (results['model'] == 'parametric') & (
                results[
                    'graph'] == 'parallel') & (
                results[
                    'telescope'] == tel)][
        'time'] / (
            512 / max_comp)
#
    # Updating the types because we are manipulating parametric to our advantage
    # Parametric values are the same regardless of parallel/base, so we can use
    # one set as an alternative selection of data
    results.loc[
        (results['model'] == 'parametric') & (results['graph'] == 'parallel'), [
            'graph', 'model']] = 'parametric_compute_adjusted'

results.loc[results['model'] == 'parametric', 'graph'] = 'parametric'


hatches = ['-', 'x', '\\']  # , '\\', '*', 'o']

# Loop over the bars

xaxis = set(results['hpso'])

initial_parametric = initial[initial['model'] =='parametric'][['hpso','time']]

g = sns.barplot(y='time', x='hpso', hue='graph', data=results)
# g.set(yscale="log")
for i, thisbar in enumerate(g.patches):
    # Set a different hatch for each bar
    thisbar.set_hatch(hatches[i % 3])

# plt.savefig(f'{str(curr_results).strip(curr_results.suffix)}_barplot.png')
