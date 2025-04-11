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

import sys
from datetime import date
from copy import deepcopy
from pathlib import Path
from multiprocessing import Pool, Manager
from shadow.models.workflow import Workflow
from shadow.models.environment import Environment
from shadow.algorithms.heuristic import heft, fcfs
from skaworkflows.config_generator import config_to_shadow
from skaworkflows.parametric_runner import \
    calculate_parametric_runtime_estimates

sys.path.append('/home/rwb/github/skaworkflows')

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
    wf, env, f, graph, telescope, position, nodes, channels, lock, pipeline_sets = tup

    if "standard" in wf.name:
        data = 'no_data'
    elif "no_data" in  wf.name and "edges" in wf.name:
        data = 'edges'
    else:
        data = 'task_edges'

    workflow = Workflow(wf)
    hpso = f.split("_")[0]   

    workflow.add_environment(env)
    print(f"Running FCFS {position} {wf.name}")
    fcfs_res = fcfs(workflow, position).makespan
    # heft_res = None
    res_str_fcfs = (f"{hpso},workflow,{fcfs_res},"
                    f"{graph},{telescope},{nodes},{channels},{data},{pipeline_sets}")
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
    # print(tup)
    wf, env, f, graph, telescope, position, nodes, channels, lock, pipeline_sets = tup
    if graph == 'scatter' or channels == 512 or "no_data" not in wf.name:
        return None
    if "no_data" in wf.name:
        data = False
    else:
        data = True
    hpso = f.split("_")[0]
    pipeline_list = None
    if pipeline_sets:
        pipeline_list = pipeline_sets.split('_')
    par_res = calculate_parametric_runtime_estimates(
        PAR_MODEL_SIZING, telescope, [hpso], pipeline_list
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


def low_setup(config_iterations, channel_iterations, wfdict, lock, pipeline_sets):
    print('Low Setup')
    data_iterations = [False] # , True]
    graph_iterations = ["prototype"] #, 'scatter']
    count = 0
    config = 896
    hpsos=['hpso01','hpso02a', 'hpso02b']
    for hpso in hpsos:
        for data in data_iterations:
            for channel in channel_iterations:
               # Set up paths to the respective directory
                for graph in graph_iterations:
                    low_path = Path(
                        f"{BASE_DIR}/low_maximal/{graph}/"
                    )
                    low_config_shadow = config_to_shadow(low_path / f"no_data_low_sdp_config_{graph}_n{config}_{channel}channels.json", 'shadow_')
                    low_env = Environment(low_config_shadow)
                    f = low_path / 'workflows' / f"{hpso}_time-18000_channels-{channel}_tel-512_no_data-standard.json"
                    fedges = low_path / 'workflows' / f"{hpso}_time-18000_channels-{channel}_tel-512_no_data-edges.json"
                    fdata = low_path / 'workflows' / f"{hpso}_time-18000_channels-{channel}_tel-512-edges.json"
                    for x in range(1):
                        nenv = deepcopy(low_env)
                        wfstr = f"{f.name}_{graph}_{x}"
                        wfstr_edges = f"{fedges.name}_{graph}_{x}"
                        wfstr_data = f"{fdata.name}_{graph}_{x}"
                        wfdict[wfstr] = (
                            f, nenv, f.name, graph, 'low-adjusted', count,
                        config, channel, lock, pipeline_sets
                        )
                        wfdict[wfstr_data] = (
                            fdata, nenv, fdata.name, graph, 'low-adjusted', count,
                        config, channel, lock, pipeline_sets
                        )
                        wfdict[wfstr_edges] = (
                            fedges, nenv, fedges.name, graph, 'low-adjusted', count,
                        config, channel, lock, pipeline_sets
                        )
                        if graph == 'scatter':
                            break
                        count += 1
    return wfdict

def mid_setup(config_iterations, channel_iterations,wfdict,lock, pipeline_sets):
    data_iterations = [False, True]
    graph_iterations = ['prototype'] #, 'cont_img_mvp_graph']
    count = 0
    config = 786
    hpsos=['hpso13','hpso15', 'hpso22', 'hpso32']
    hpso_times = {'hpso13': 28800 ,'hpso15':15840 , 'hpso22':28800, 'hpso32':7920}
    for hpso in hpsos:
        for data in data_iterations:
            for channel in channel_iterations:
               # Set up paths to the respective directory
                for graph in graph_iterations:
                    mid_path = Path(
                        f"{BASE_DIR}/mid_maximal/{graph}/"
                    )
                    mid_config_shadow = config_to_shadow(mid_path / f"no_data_mid_sdp_config_{graph}_n{config}_{channel}channels.json", 'shadow_')
                    mid_env = Environment(mid_config_shadow)
                    f = mid_path / 'workflows' / f"{hpso}_time-{hpso_times[hpso]}_channels-{channel}_tel-197_no_data-standard.json"
                    fedges = mid_path / 'workflows' / f"{hpso}_time-{hpso_times[hpso]}_channels-{channel}_tel-197_no_data-edges.json"
                    fdata = mid_path / 'workflows' / f"{hpso}_time-{hpso_times[hpso]}_channels-{channel}_tel-197-edges.json"
                    nenv = deepcopy(mid_env)
                    for x in range(1):
                        wfstr = f"{f.name}_{graph}_{x}"
                        wfstr_edges = f"{fedges.name}_{graph}_{x}"
                        wfstr_data = f"{fdata.name}_{graph}_{x}"
                        wfdict[wfstr] = (
                            f, nenv, f.name, graph, 'mid-adjusted', count,
                        config, channel, lock, pipeline_sets
                        )
                        wfdict[wfstr_data] = (
                            fdata, nenv, fdata.name, graph, 'mid-adjusted', count,
                        config, channel,lock, pipeline_sets
                        )
                        wfdict[wfstr_edges] = (
                            fedges, nenv, fedges.name, graph, 'mid-adjusted', count,
                        config, channel,lock, pipeline_sets
                        )

                        if graph == 'scatter':
                            break
                        count += 1
    return wfdict

if __name__ == '__main__':
    BASE_DIR = Path(f"/home/rwb/Dropbox/University/PhD/experiment_data/chapter3/initial_results/")
    wfs_dict = {}
    low_config_iterations = [896] # 512]
    low_channel_iterations = [896] # , 512]
    manager = Manager()
    queue = manager.Queue()
    lock = manager.Lock()
    pipeline_sets = None # None = Maximal use case with all workflows

    wfs_dict = low_setup(low_config_iterations, low_channel_iterations, wfs_dict, lock, pipeline_sets)
    mid_config_iterations = [786] #, 512]
    mid_channel_iterations = [786] # , 512]
    wfs_dict = mid_setup(mid_config_iterations, mid_channel_iterations, wfs_dict, lock, pipeline_sets)
    # for e in sorted(wfs_dict.keys()):
    #     print(e)
    output = Path(
        f"chapter3/initial_results/results_{date.today().isoformat()}.csv")
    if not output.exists:
        with output.open('w+') as fo:
            fo.write('hpso,workflow,time,graph,telescope,nodes,channels,data,pipeline_sets')

    params = set(zip(wfs_dict.values(), [queue for x in range(len(wfs_dict))]))

    with Pool(processes=1) as pool:
        result = pool.starmap(run_parametric, params)
    with Pool(processes=4) as pool:
        result = pool.starmap(run_shadow, params)
