# Copyright (C) 3/2/21 RW Bunney

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

import simpy
import logging

from topsim.core.simulation import Simulation
from topsim.core.delay import DelayModel
from user.telescope import Telescope
from user.scheduling import GreedyAlgorithmFromPlan

logging.basicConfig(level="DEBUG")
LOGGER = logging.getLogger(__name__)

CONFIG = 'multi_observation_simulation.json'

env = simpy.Environment()
dm = DelayModel(0.25, 'normal', DelayModel.DelayDegree.HIGH)

instrument = Telescope
simulation = Simulation(
    env=env,
    config=CONFIG,
    instrument=instrument,
    algorithm_map={'heft': 'heft', 'fifo': GreedyAlgorithmFromPlan},
    delay=dm
)

# LOGGER.info("Simulation Starting")
simulation.start(-1)
# simulation.resume(10000)
