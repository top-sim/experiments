# Copyright (C) 13/8/23 RW Bunney

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
import time
import logging
import simpy

sys.path.insert(0, os.path.abspath('../..'))
sys.path.append("/home/rwb/github/skaworkflows")
sys.path.insert(0, os.path.abspath('../../shadow'))

from pathlib import Path
from datetime import date
# Framework defined models
from topsim.core.simulation import Simulation
from topsim.core.delay import DelayModel

# User defined models
from topsim_user.telescope import Telescope  # Instrument
from topsim_user.schedule.batch_allocation import BatchProcessing
from topsim_user.plan.batch_planning import BatchPlanning  # Planning
from topsim_user.plan.static_planning import SHADOWPlanning
from topsim_user.schedule.dynamic_plan import DynamicSchedulingFromPlan

logging.basicConfig(level="INFO")
LOGGER = logging.getLogger(__name__)

RUN_PATH = Path.cwd()
FOLDER_PATH = Path('chapter4/motivating_example_topsim')

simulation_configuration = Path(
    '/home/rwb/Dropbox/University/PhD/experiment_data/chapter4/'
    + 'motivating_example/heft_single_observation_simulation.json')

if __name__ == '__main__':
    st = time.time()
    LOGGER.info(f"Running experiment from {RUN_PATH}/{FOLDER_PATH}")
    env = simpy.Environment()
    instrument = Telescope
    simulation = Simulation(
        env=env, config=simulation_configuration,
        instrument=instrument, planning_algorithm='batch',
        planning_model=BatchPlanning('batch'),
        scheduling=BatchProcessing(
            min_resources_per_workflow=1,
            resource_split={'emu': (3, 3)},
            max_resource_partitions=1
        ),
        delay=None, timestamp=None, to_file=True,
        hdf5_path=f'{RUN_PATH}/{FOLDER_PATH}/results_'
                  f'{date.today().isoformat()}.h5', )
    simulation.start()
    ft = time.time()
    LOGGER.info(f"Experiment took {(ft-st)/60} minutes to run")
