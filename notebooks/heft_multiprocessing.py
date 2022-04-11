# Copyright (C) 8/4/22 RW Bunney

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


import time
from copy import deepcopy
from pathlib import Path
from multiprocessing import Pool
from shadow.models.workflow import Workflow
from shadow.models.environment import Environment
from shadow.algorithms.heuristic import heft


def run_workflow(workflow, environment):
    workflow.add_environment(environment)
    return heft(workflow)


if __name__ == '__main__':

    output_dir_path = Path(f"notebooks/sdp_comparison/output")
    if not  output_dir_path.exists():
        raise RuntimeError(f"{output_dir_path} does not exist!")

    config_path = output_dir_path / "shadow_config.json"

    wfs = []
    for wf_path in (output_dir_path / "workflows").iterdir():
        wfs.append(Workflow(wf_path))

    env = Environment(config_path)
    environs = [deepcopy(env) for i in range(3)]

    params = list(zip(wfs, environs))
    start = time.time()

    with Pool(processes=4) as pool:
        result = pool.starmap(run_workflow, params)
    finish = time.time()
    print(f"{finish-start=}")

    # start = time.time()
    # for x in range(3):
    #     workflow = Workflow("/home/rwb/github/thesis_experiments/skaworkflows_tests"
    #                  "/workflows/hpso01_time-3600_channels-256_tel-512.json")
    #     workflow.add_environment(env)
    #     print(heft(workflow))
    # finish = time.time()
    # print(f"{finish-start=}")
