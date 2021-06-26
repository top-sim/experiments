# Copyright (C) 18/5/21 RW Bunney

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
import logging
import time
import simpy

import pandas as pd

sys.path.insert(0, os.path.abspath('../../thesis_experiments'))
sys.path.insert(0, os.path.abspath('../../topsim_pipelines'))

from topsim.core.simulation import Simulation
from topsim.core.delay import DelayModel
from user.telescope import Telescope
from user.scheduling import DynamicAlgorithmFromPlan

logging.basicConfig(level="DEBUG")
logger = logging.getLogger()

algorithms = ['heft', 'fcfs']
global_sim = pd.DataFrame()
global_tasks = pd.DataFrame()

# RUNNING SIMULATION AND GENERATING DATA

for algorithm in algorithms:
    for config in sorted(os.listdir(
            '2021_isc-hpc/config/single_size/40cluster')):
        if '.json' in config:
            CONFIG = f'2021_isc-hpc/config/single_size/40cluster/{config}'
            env = simpy.Environment()
            instrument = Telescope
            timestamp = f'{time.time()}'.split('.')[0]
            simulation = Simulation(
                env=env,
                config=CONFIG,
                instrument=instrument,
                planning=algorithm,
                scheduling=DynamicAlgorithmFromPlan,
                delay=None,
                timestamp={timestamp}
            )
            sim, tasks = simulation.start()
            global_sim = global_sim.append(sim)
            global_tasks = global_tasks.append(tasks)
            print(algorithm, config, len(sim))
global_tasks.to_pickle('tasks_output.pkl')
global_sim.to_pickle('simulation_output.pkl')

# PLOTTING SIMULATION DATA - originally produced in a Jupyter Notebook

# Group by planning, delay, and config to get the simulation time for each
# simulation.
df = global_sim.groupby(['planning','delay', 'config']).size().astype(float).reset_index(name='time').sort_values(by=['planning'])
df['config'] = df['config'].str.replace('visualisation_playground/sim_config/single_size/40cluster/mos_sw','').str.strip('.json').astype(float)
basetime = pd.Series(df[df['planning'] == 'fcfs']['time'])
basetime = basetime.append(basetime,ignore_index=True)
df['increase'] = basetime/df['time']

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(8,4))
plot = sns.barplot(data=df,x='config', y='increase', hue='planning', palette=['red','blue'],
axes=ax)
h, l = ax.get_legend_handles_labels()
ax.legend(h,['FCFS','HEFT'], title='Static planning strategy',loc='upper right')
ax.set_ylabel('% Improvement on final schedule Makespan)')
ax.set_xlabel('Max Workflow Parallelism')
ax.set(ylim=(0.9,1.2))
fig.savefig('ratio_comparison.png', format="png")

