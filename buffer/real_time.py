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

from topsim.core.simulation import Simulation
from user.telescope import Telescope
from user.scheduling import FifoAlgorithm


EVENT_FILE = 'buffer/config/real_time.trace'
CONFIG = 'buffer/config/real_time.json'

env = simpy.Environment()

planning_algorithm = 'heft'
scheduling_algorithm = FifoAlgorithm
instrument = Telescope

simulation = Simulation(
    env=env,
    config=CONFIG,
    instrument=instrument,
    algorithm_map={'pheft': 'pheft', 'heft': 'heft', 'fifo': FifoAlgorithm},
    event_file=EVENT_FILE,
)

simulation.start(11)
simulation.resume(10000)
