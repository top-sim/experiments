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


def run_simulation(boundaries):
    """
    Given the specified, construct the simulation object and run it

    """
    pass


if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath('../../thesis_experiments'))
    sys.path.insert(0, os.path.abspath('../../topsim_pipelines'))
    sys.path.insert(0, os.path.abspath('../../shadow'))
    os.chdir("/home/rwb/github/thesis_experiments/")
    wf_config = (
        '2021_isc-hpc/config/single_size/40cluster/mos_sw80.json'
    )
    OUTPUT_DIR = 'simulation_output/batch_allocation_experiments'
    run_simulation(None)
