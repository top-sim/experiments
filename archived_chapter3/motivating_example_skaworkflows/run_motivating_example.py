# Copyright (C) 10/6/23 RW Bunney

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

heft_env = Path('/home/rwb/Dropbox/University/PhD/experiment_data/chapter3/motivating_example/final_heft_sys.json')
heft_wf = Path('/home/rwb/Dropbox/University/PhD/experiment_data/chapter3/motivating_example/final_heft.json')
heft_wf_data = Path('/home/rwb/Dropbox/University/PhD/experiment_data/chapter3/motivating_example/final_heft_somedata.json')
low_env = Environment(heft_env)
workflow = Workflow(heft_wf)
workflow.add_environment(low_env)

total_resources = sum([x.flops for x in low_env.machines])
total_compute = sum([t.flops_demand for t in workflow.graph.nodes])
print(f"{total_resources=}", f"{total_compute=}")
print(f"Compute/Resources = {total_compute/total_resources}")

fcfs_res = fcfs(workflow, 0).makespan
heft_res = heft(workflow, 1).makespan
workflow_data = Workflow(heft_wf_data)
workflow_data.add_environment(low_env)
fcfs_data = fcfs(workflow_data, 0).makespan
heft_data = heft(workflow_data, 1).makespan

print(f"FCFS {fcfs_res}")
print(f"HEFT {heft_res}")
print(f"FCFS (with data) {fcfs_data}")
print(f"HEFT (with data) {heft_data}")
