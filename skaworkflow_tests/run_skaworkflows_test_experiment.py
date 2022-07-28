# Copyright (C) 22/2/22 RW Bunney

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

sys.path.insert(0, os.path.abspath('../../thesis_experiments'))
sys.path.insert(0, os.path.abspath('../../skaworkflows'))
sys.path.insert(0, os.path.abspath('../../shadow'))

import logging
import simpy
from pathlib import Path
from datetime import date

import skaworkflows.workflow.hpso_to_observation as hto
from skaworkflows.config_generator import create_config
from skaworkflows.hpconfig.specs.sdp import SDP_LOW_CDR
from skaworkflows import common

logging.basicConfig(level="INFO")
LOGGER = logging.getLogger(__name__)

RUN_PATH = Path.cwd()
FOLDER_PATH = Path(f'skaworkflow_tests')

cfg_path = Path('skaworkflow_tests/low_parallel/low_sdp_config.json')

if not cfg_path.exists():
    LOGGER.info(f"Exiting simulation, simulation config does not exist")

sys.path.insert(0, os.path.abspath('../../thesis_experiments'))
sys.path.insert(0, os.path.abspath('../../topsim_pipelines'))
sys.path.insert(0, os.path.abspath('../../shadow'))

# Framework defined models
from topsim.core.simulation import Simulation
from topsim.core.delay import DelayModel

# User defined models
from topsim_user.telescope import Telescope  # Instrument
# from topsim_user.schedule.dynamic_plan import DynamicAlgorithmFromPlan  # Scheduling
from topsim_user.schedule.greedy import GreedyAlgorithmFromPlan  # Scheduling
from topsim_user.schedule.batch_allocation import BatchProcessing
from topsim_user.plan.batch_planning import BatchPlanning  # Planning
# from topsim_user.plan.static_planning import SHADOWPlanning

if __name__ == '__main__':

    LOGGER.info(f"Running experiment from {RUN_PATH}/{FOLDER_PATH}")
    env = simpy.Environment()
    instrument = Telescope
    # timestamp = f'{time.time()}'.split('.')[0]
    simulation = Simulation(
        env=env,
        config=cfg_path,
        instrument=instrument,
        planning_algorithm='batch',
        planning_model=BatchPlanning('batch'),
        scheduling=BatchProcessing,
        delay=None,
        timestamp='skaworkflow_test',
        to_file=True,
        hdf5_path=f'{RUN_PATH}/{FOLDER_PATH}/results_'
                  f'{date.today().isoformat()}.h5',

        # hdf5_path='',
        # delimiters=f'test/'
    )
    simulation.start()
LOGGER.info(f"Experiment finished, exiting script...")
