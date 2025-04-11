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

# Necessary models

import os
import sys
import logging
import datetime
import simpy
import pandas as pd

from topsim.core.simulation import Simulation
from topsim_user.schedule.batch_allocation import BatchProcessing
from topsim_user.telescope import Telescope  # Instrument
from topsim_user.plan.batch_planning import BatchPlanning  # Planning


def init():
    """
    Initialise a simulation object

    Returns
    -------
    sim : simulation objects
    """
    pass


def concatenate_pickle_file(dir):
    """
    Given a directory of pickle files, concatenate them to create a single
    file of multiple simulations
    Parameters
    ----------
    dir

    Returns
    -------
    pkl
    """
    pass


def run_simulation(cfg, timestamp):
    """
    Given the specified, construct the simulation object and run it

    """
    env = simpy.Environment()
    simulation = Simulation(
        env=env,
        config=cfg,
        instrument=Telescope,
        planning_algorithm='batch',
        planning_model=BatchPlanning('batch'),
        scheduling=BatchProcessing,
        delay=None,
        timestamp=timestamp,
        to_file=True
    )
    sim, tasks = simulation.start()
    return sim, tasks


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    logger = logging.getLogger()

    sys.path.insert(0, os.path.abspath('../../thesis_experiments'))
    sys.path.insert(0, os.path.abspath('../../topsim_pipelines'))
    sys.path.insert(0, os.path.abspath('../../shadow'))
    os.chdir("/home/rwb/github/thesis_experiments/")
    CONFIG = (
        'publications/2021_isc-hpc/config/single_size/40cluster/mos_sw80.json'
    )
    OUTPUT_DIR = 'simulation_output/batch_allocation_experiments'

    STATIC_PLANNING_ALGORITHMS = ['heft', 'fcfs']

    GLOBAL_SIM = pd.DataFrame()
    GLOBAL_TASKS = pd.DataFrame()

    DATE = datetime.date.today().strftime("%Y-%m-%d")
    FNAME = f"{OUTPUT_DIR}/{DATE}"

    _sim, _tasks = run_simulation(CONFIG, timestamp=FNAME)
