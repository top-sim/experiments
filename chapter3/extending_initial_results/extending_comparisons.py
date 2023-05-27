# Copyright (C) 27/5/23 RW Bunney

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

from pathlib import Path
from shadow.models.workflow import Workflow
from shadow.models.environment import Environment
from shadow.algorithms.heuristic import heft, fcfs

import sys
sys.path.append('/home/rwb/github/skaworkflows')

from skaworkflows.config_generator import config_to_shadow

low_config_shadow = Path("chapter3/extending_initial_results/shadow_no_data_low_sdp_config_prototype_n896_896channels.json")

low_env = Environment(low_config_shadow)

wf = Path("chapter3/extending_initial_results/workflows/hpso01_time-18000_channels-896_tel-512.json")

workflow = Workflow(wf)

workflow.add_environment(low_env)
fcfs_res = fcfs(workflow).makespan
print(fcfs_res)
with open("chapter3/extending_initial_results/extending_comparisons.csv", 'w+') as f:
    f.write(f"hpso01,workflow,{fcfs_res},prototype_no_transfer,low-adjusted,896,512,True")
