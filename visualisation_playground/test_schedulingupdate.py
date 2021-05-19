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
from user.scheduling import GreedyAlgorithmFromPlan

logging.basicConfig(level="INFO")
logger = logging.getLogger()

algorithms = ['heft', 'fcfs']
global_sim = pd.DataFrame()
global_tasks = pd.DataFrame()
# for dm in dm_list:
for algorithm in algorithms:
    for config in sorted(os.listdir('visualisation_playground/sim_config/single_size/40cluster')):
        if '.json' in config:
            CONFIG = f'visualisation_playground/sim_config/single_size/40cluster/{config}'
            env = simpy.Environment()
            instrument = Telescope
            timestamp = f'{time.time()}'.split('.')[0]
            simulation = Simulation(
                env=env,
                config=CONFIG,
                instrument=instrument,
                planning=algorithm,
                scheduling=GreedyAlgorithmFromPlan,
                delay=None,
                timestamp={timestamp}
            )
            sim, tasks = simulation.start()
            global_sim = global_sim.append(sim)
            global_tasks = global_tasks.append(tasks)
            print(algorithm, config, len(sim))
global_tasks.to_pickle('hopefully_final_scheduling_update.pkl')
global_sim.to_pickle('hopefully_final_scheduling_update.pkl')
