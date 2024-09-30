# Copyright (C) 30/8/23 RW Bunney

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

import sys
from pathlib import Path
sys.path.append('/home/rwb/github/skaworkflows')

from shadow.models.workflow import Workflow
from shadow.models.environment import Environment
from shadow.algorithms.heuristic import heft, fcfs

from skaworkflows.config_generator import config_to_shadow
from skaworkflows.parametric_runner import \
    calculate_parametric_runtime_estimates

BASEDIR = Path("/home/rwb/Dropbox/University/PhD/experiment_data/chapter4/scalability/low/prototype/")
config = BASEDIR  / "no_data_low_sdp_config_prototype_channels.json"
wfpath = BASEDIR / 'workflows' / "hpso01_time-18000_channels-512_tel-32_no_data.json"

low_config_shadow = config_to_shadow(config,"shadow")

workflow = Workflow(wfpath)
# hpso = f.name.split('_')[0]
low_env = Environment(low_config_shadow)
workflow.add_environment(low_env)
fcfs_res = heft(workflow, 0).makespan
print(f"Makespan {fcfs_res}")