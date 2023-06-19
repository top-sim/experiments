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


from datetime import date
from copy import deepcopy
from pathlib import Path
from multiprocessing import Pool, Manager
from shadow.models.workflow import Workflow
from shadow.models.environment import Environment
from shadow.algorithms.heuristic import heft, fcfs

import sys
sys.path.append('/home/rwb/github/skaworkflows')

from skaworkflows.config_generator import config_to_shadow
from skaworkflows.parametric_runner import \
    calculate_parametric_runtime_estimates

# from chapter5.parametric_model_baselines.generate_data import (
#     LOW_HPSO_PATHS, MID_HPSO_PATHS)

PAR_MODEL_SIZING = Path("../sdp-par-model/2021-06-02_LongBaseline_HPSOs.csv")


def run_workflow(workflow, environment):
    workflow.add_environment(environment)
    return heft(workflow)


def run_shadow(tup, queue, test=False):
    if test:
        print(f"{tup}")
        return None
    wf, env, f, graph, telescope, position, nodes, channels, lock = tup
    if "no_data" in wf.name:
        data = False
    else:
        data = True
    workflow = Workflow(wf)
    hpso = f.split("_")[0]   

    workflow.add_environment(env)
    print(f"Running FCFS {position}")
    fcfs_res = fcfs(workflow, position).makespan
    # heft_res = None
    res_str_fcfs = (f"{hpso},workflow,{fcfs_res},"
                    f"{graph},{telescope},{nodes},{channels},{data}")
    print(res_str_fcfs)
    lock.acquire()
    with output.open('a') as f:
        f.write(f"{str(res_str_fcfs)}\n")
        f.flush()
    lock.release()


def run_parametric(tup, queue, test=False):
    """
    Calculate the runtime estimates from the parametric model
    We only need one of these values for each workflow, so skip
    one of the graph types.

    Parameters
    ----------
    tup : tuple containing relevant workflows and information for this instance
    test : True if we are verifying the data in this function; false if we
    are running this for real.

    Returns
    -------

    """
    if test:
        return None

    wf, env, f, graph, telescope, position, nodes, channels, lock = tup
    if graph == 'scatter' or channels == 512 or "no_data" not in wf.name:
        return None
    if "no_data" in wf.name:
        data = False
    else:
        data = True
    hpso = f.split("_")[0]
    par_res = calculate_parametric_runtime_estimates(
        PAR_MODEL_SIZING, telescope, [hpso]
    )
    res_str = (
        f"{hpso},parametric,{par_res[hpso]['time']},"
        f"{graph},{telescope},{nodes},{channels},{data}"
    )
    print(res_str)
    lock.acquire()
    with output.open('a') as f:
        f.write(f"{str(res_str)}\n")
        f.flush()
    lock.release()


def run_test(tup, queue, test=True):
    wf, env, f, graph, telescope, position, nodes, channels, lock = tup
    lock.acquire()
    with output.open('a') as f:
        f.write(f"{str(position)}\n")
    lock.release()


def listener(queue, output: Path):
    """Listen for messages on the queue and write updates to the XML log."""

    with output.open('a') as f:
        while True:
            msg = queue.get()
            if str(msg) == 'Finish':
                # print(msg)
                break
            f.write(str(msg))
            f.flush()


def low_setup(config_iterations, channel_iterations, wfdict, lock):
    data_iterations = [False, True]
    graph_iterations = ['prototype', 'scatter']
    count = 0
    config = 896
    hpsos=['hpso01','hpso02a', 'hpso2b']
    for hpso in hpsos:
        for data in data_iterations:
            for channel in channel_iterations:
               # Set up paths to the respective directory
                for graph in graph_iterations:
                    low_path = Path(
                        f"{BASE_DIR}/low/{graph}/"
                    )
                    low_config_shadow = config_to_shadow(low_path / f"no_data_low_sdp_config_{graph}_n{config}_{channel}channels.json", 'shadow_')
                    low_env = Environment(low_config_shadow)
                    f = low_path / 'workflows' / f"{hpso}_time-18000_channels-{channel}_tel-512_no_data.json"
                    fdata = low_path / 'workflows' / f"{hpso}_time-18000_channels-{channel}_tel-512.json"
                    nenv = deepcopy(low_env)
                    for x in range(10):
                        wfstr = f"{f.name}_{graph}_{x}"
                        wfstr_data = f"{fdata.name}_{graph}_{x}"
                        wfdict[wfstr] = (
                            f, nenv, f.name, graph, 'low-adjusted', count,
                        config, channel,lock
                        )
                        wfdict[wfstr_data] = (
                            fdata, nenv, fdata.name, graph, 'low-adjusted', count,
                        config, channel,lock
                        )
                        if graph == 'scatter':
                            break
                        count += 1
    return wfdict

def mid_setup(config_iterations, channel_iterations,wfdict,lock):
    data_iterations = [False, True]
    graph_iterations = ['prototype', 'scatter']
    count = 0
    config = 896
    for data in data_iterations:
        for channel in channel_iterations:
            # Set up paths to the respective directory
            if config == 512 and channel == 896:
                # We are not interested in this experiment
                continue
            for graph in graph_iterations:
                mid_path = Path(
                    f"{BASE_DIR}/mid/{graph}/"
                )
                mid_config_shadow = config_to_shadow(mid_path / f"no_data_mid_sdp_config_{graph}_n{config}_{channel}channels.json", 'shadow_')
                mid_env = Environment(mid_config_shadow)
                for f in (Path(mid_path) / 'workflows').iterdir():
                    nenv = deepcopy(mid_env)
                    wfstr = (
                        f"{f.name}{graph}mid-adjusted{config}"
                        f"{channel}")
                    if wfstr in wfdict:
                        continue
                    else:
                        wfdict[wfstr] = (
                        f, nenv, f.name, graph, 'mid-adjusted', count,
                        config, channel,lock)
                    count += 1

    return wfdict


if __name__ == '__main__':
    BASE_DIR = Path(f"chapter3/")
    wfs_dict = {}
    low_config_iterations = [896] # 512]
    low_channel_iterations = [896] # , 512]
    manager = Manager()
    queue = manager.Queue()
    lock = manager.Lock()

    wfs_dict = low_setup(low_config_iterations, low_channel_iterations, wfs_dict, lock)
    # mid_config_iterations = [786] #, 512]
    # mid_channel_iterations = [896] # , 512]
    # wfs_dict = mid_setup(mid_config_iterations, mid_channel_iterations, wfs_dict, lock)
    # for e in sorted(wfs_dict.keys()):
    #     print(e)
    output = Path(
        f"chapter3/initial_results/results_{date.today().isoformat()}.csv")
    with output.open('w+') as fo:
        fo.write('hpso,workflow,time,graph,telescope,nodes,channels,data')

    params = set(zip(wfs_dict.values(), [queue for x in range(len(wfs_dict))]))
    # with Pool(processes=1) as pool:
    #     result = pool.starmap(run_parametric, params)
    with Pool(processes=4) as pool:
        result = pool.starmap(run_shadow, params)
