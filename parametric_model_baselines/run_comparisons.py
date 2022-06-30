# Copyright (C) 9/5/22 RW Bunney

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
from multiprocessing import Pool, Manager
from shadow.models.workflow import Workflow
from shadow.models.environment import Environment
from shadow.algorithms.heuristic import heft, fcfs

from skaworkflows.config_generator import config_to_shadow
from skaworkflows.parametric_runner import \
    calculate_parametric_runtime_estimates

from parametric_model_baselines.generate_data import (
    LOW_OUTPUT_DIR, LOW_OUTPUT_DIR_PAR, MID_OUTPUT_DIR, MID_OUTPUT_DIR_PAR,
    LOW_TOTAL_SIZING, MID_TOTAL_SIZING
)

PAR_MODEL_SIZING = Path(
    "../sdp-par-model/2021-06-02_LongBaseline_HPSOs.csv"
)


def run_workflow(workflow, environment):
    workflow.add_environment(environment)
    return heft(workflow)


def run_mulitple():
    env = Environment("/home/rwb/github/thesis_experiments/"
                      "notebooks/sdp_comparison/output/shadow_config.json")
    # files = list(wf_path.iterdir())
    wfs = [
        Workflow("/home/rwb/github/thesis_experiments/skaworkflows_tests"
                 "/workflows/hpso01_time-3600_channels-256_tel-512.json") for i
        in range(3)
    ]
    environs = [deepcopy(env) for i in range(3)]

    params = list(zip(wfs, environs))
    start = time.time()
    with Pool(processes=4) as pool:
        result = pool.starmap(run_workflow, params)
    finish = time.time()
    print(f"{finish-start=}")

def run_par_runner(tup, queue):
    wf, env, f, graph, telescope,position = tup
    workflow = Workflow(wf)
    hpso = f.split("_")[0]
    par_res = calculate_parametric_runtime_estimates(PAR_MODEL_SIZING,
                                                     telescope,
                                                     [hpso])
    res_str = (
        f"{hpso},parametric,{par_res[hpso]['time']},"
        f"{graph},{telescope}\n"
    )
    queue.put(res_str)

    workflow.add_environment(env)
    heft_res = heft(workflow, position).makespan

    res_str_heft = (
        f"{hpso},workflow,{heft_res},"
        f"{graph},{telescope}\n"
    )
    queue.put(res_str_heft)



def listener(queue, output: Path):
    """Listen for messages on the queue and write updates to the XML log."""

    with output.open('w') as f:
        while True:
            msg = queue.get()
            # print(msg)
            if str(msg) == 'Finish':
                # print(msg)
                break
            f.write(str(msg))
            f.flush()


if __name__ == '__main__':
    low_base_dir = Path("parametric_model_baselines/low_base")
    low_base_workflows_dir = low_base_dir / "workflows"
    low_config_shadow = config_to_shadow(low_base_dir /
                                         "low_sdp_config_512nodes.json",
                                         'shadow_')
    low_par_dir = Path("parametric_model_baselines/low_parallel")
    low_par_workflows_dir = low_par_dir / "workflows"

    low_wfs = []
    low_env = Environment(low_config_shadow)
    count=0
    for i,f in enumerate(low_base_workflows_dir.iterdir()):
        nenv = deepcopy(low_env)
        low_wfs.append((f, nenv, f.name, 'base', 'low-adjusted',count))
        count += 1
    for i,f in enumerate(low_par_workflows_dir.iterdir()):
        nenv = deepcopy(low_env)
        low_wfs.append((f, nenv, f.name, 'parallel', 'low-adjusted', count))
        count+=1

    mid_base_dir = Path("parametric_model_baselines/mid_base")
    mid_base_workflows_dir = mid_base_dir / "workflows"
    mid_config_shadow = config_to_shadow(
        mid_base_dir / "mid_sdp_config_512nodes.json", 'shadow_'
    )
    mid_par_dir = Path("parametric_model_baselines/mid_parallel")
    mid_par_workflows_dir = mid_par_dir / "workflows"

    mid_env = Environment(mid_config_shadow)
    mid_wfs = []
    for f in mid_base_workflows_dir.iterdir():
        nenv = deepcopy(mid_env)
        mid_wfs.append((f, nenv, f.name, 'base', 'mid-adjusted',count))
        count+=1
    for f in mid_par_workflows_dir.iterdir():
        nenv = deepcopy(mid_env)
        mid_wfs.append((f, nenv, f.name, 'parallel', 'mid-adjusted',count))
        count+=1

    # Parametric params
    manager = Manager()
    queue = manager.Queue()
    output = Path("parametric_model_baselines/output.txt")
    wfs = low_wfs + mid_wfs
    params = list(zip(wfs, [queue for x in range(len(wfs))]))
    for thruple in params:
        print(thruple)  # help determine which order should be in file
    with Pool(processes=6) as pool:
        lstn = pool.apply_async(listener, (queue, output))
        result = pool.starmap(run_par_runner, params)

        queue.put('Finish')
    # for pair in low_wfs:
    #     wf, fname = pair
    #     hpso = fname.split("_")[0]
    #     res = calculate_parametric_runtime_estimates(PAR_MODEL_LOW_SIZING,
    #                                                  low_scenario, [hpso])
    #     print(res[hpso]['time'])
