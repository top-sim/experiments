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
import time

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../../skaworkflows'))
sys.path.insert(0, os.path.abspath('../../../shadow'))

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
FOLDER_PATH = Path(f'chapter6/skaworkflow_tests')

cfg_path = Path(
    'chapter6/skaworkflow_tests/low_parallel/low_sdp_config_dual.json')

if not cfg_path.exists():
    LOGGER.info(f"Exiting simulation, simulation config does not exist")

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../../topsim_pipelines'))
sys.path.insert(0, os.path.abspath('../../../shadow'))

# Framework defined models
from topsim.core.simulation import Simulation
from topsim.core.delay import DelayModel

# User defined models
from topsim_user.telescope import Telescope  # Instrument
from topsim_user.schedule.batch_allocation import BatchProcessing
from topsim_user.plan.batch_planning import BatchPlanning  # Planning

if __name__ == '__main__':
    st = time.time()
    LOGGER.info(f"Running experiment from {RUN_PATH}/{FOLDER_PATH}")
    env = simpy.Environment()
    instrument = Telescope
    simulation = Simulation(env=env, config=cfg_path, instrument=instrument,
        planning_algorithm='batch', planning_model=BatchPlanning('batch'),
        scheduling=BatchProcessing(min_resources_per_workflow=1,
                                   resource_split={'hpso01_0': (820, 896),
                                                   'hpso01_1':(820, 896)},
                                   max_resource_partitions=2),
     delay=None, timestamp=None, to_file=True,
        hdf5_path=f'{RUN_PATH}/{FOLDER_PATH}/results_'
                  f'{date.today().isoformat()}.h5', )
    simulation.start()
    ft = time.time()
    LOGGER.info(f"Experiment took {(ft-st)/60} minutes to run")

LOGGER.info(f"Experiment finished, exiting script...")
