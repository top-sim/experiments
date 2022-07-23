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

from skaworkflows.config_generator import config_to_shadow
from skaworkflows.parametric_runner import \
    calculate_parametric_runtime_estimates

from parametric_model_baselines.generate_data import (LOW_HPSO_PATHS,
                                                      MID_HPSO_PATHS)

from parametric_model_baselines.generate_data import (LOW_TOTAL_SIZING,
                                                      MID_TOTAL_SIZING)

PAR_MODEL_SIZING = Path("../sdp-par-model/2021-06-02_LongBaseline_HPSOs.csv")


def run_workflow(workflow, environment):
    workflow.add_environment(environment)
    return heft(workflow)


def run_shadow(tup, queue, test=False):
    if test:
        print(f"{tup}")
        return None
    wf, env, f, graph, telescope, position, nodes, channels = tup
    if "no_data" in wf.name:
        data = False
    else:
        data = True
    workflow = Workflow(wf)
    hpso = f.split("_")[0]

    workflow.add_environment(env)
    print("Running FCFS")
    heft_res = fcfs(workflow, position).makespan
    # heft_res = None
    res_str_fcfs = (f"{hpso},workflow,{heft_res},"
                    f"{graph},{telescope},{nodes},{channels},{data}\n")
    print(res_str_fcfs)
    queue.put(res_str_fcfs)


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
        print(f"{tup}")
        return None

    wf, env, f, graph, telescope, position, nodes, channels = tup
    if graph == 'scatter':
        return None
    if "no_data" in wf.name:
        data = False
    else:
        data = True
    hpso = f.split("_")[0]
    par_res = calculate_parametric_runtime_estimates(PAR_MODEL_SIZING,
        telescope, [hpso])
    res_str = (f"{hpso},parametric,{par_res[hpso]['time']},"
               f"{graph},{telescope},{nodes},{channels},{data}\n")
    queue.put(res_str)


def listener(queue, output: Path):
    """Listen for messages on the queue and write updates to the XML log."""

    with output.open('a') as f:
        while True:
            msg = queue.get()
            # print(msg)
            if str(msg) == 'Finish':
                # print(msg)
                break
            f.write(str(msg))
            f.flush()


if __name__ == '__main__':
    BASE_DIR = Path(f"parametric_model_baselines")

    config_iterations = [896, 512]
    channel_iterations = [896, 512]
    data_iterations = [False, True]
    graph_iterations = ['prototype', 'scatter']
    wfs = []
    wfs_dict = {}
    # mid_wfs = []
    count = 0
    for data in data_iterations:
        for config in config_iterations:
            for channel in channel_iterations:
                if config == 512 and channel == 896:
                    # We are not interested in this experiment
                    continue
                low_hpso_str = next(
                    x for x in LOW_HPSO_PATHS if f'{channel}' in x)
                mid_hpso_str = next(
                    x for x in MID_HPSO_PATHS if f'{channel}' in x)
                # Set up paths to the respective directory
                for graph in graph_iterations:
                    low_path_str = (
                        f'{BASE_DIR}/low_{graph}/c{config}/n{channel}')
                    mid_path_str = (
                        f'{BASE_DIR}/mid_{graph}/c{config}/n{channel}')
                    low_config_shadow = config_to_shadow(
                        Path(f'{low_path_str}/low_sdp_config_n{config}'),
                        'shadow_')
                    mid_config_shadow = config_to_shadow(
                        Path(f'{mid_path_str}/mid_sdp_config_n{config}'),
                        'shadow_')

                    low_env = Environment(low_config_shadow)
                    mid_env = Environment(mid_config_shadow)
                    for f in (Path(low_path_str) / 'workflows').iterdir():
                        nenv = deepcopy(low_env)
                        wfstr = (
                            f"{f.name}{graph}low-adjusted{config}"
                            f"{channel}")
                        if wfstr in wfs_dict:
                            continue
                        else:
                            wfs_dict[wfstr] = (
                            f, nenv, f.name, graph, 'low-adjusted', count,
                            config, channel)

                        wfs.append(
                            (f, nenv, f.name, graph, 'low-adjusted', count,
                             config, channel))
                        count += 1
                    for f in (Path(mid_path_str) / 'workflows').iterdir():
                        nenv = deepcopy(mid_env)
                        wfstr = (
                            f"{f.name}{graph}mid-adjusted{config}"
                            f"{channel}")
                        if wfstr in wfs_dict:
                            continue
                        else:
                            wfs_dict[wfstr] = (
                            f, nenv, f.name, graph, 'mid-adjusted', count,
                            config, channel)
                        wfs.append((f, nenv, f.name, graph, 'mid-adjusted',
                                    count, config, channel))
                        count += 1

    manager = Manager()
    queue = manager.Queue()

    output = Path(
        f"parametric_model_baselines/results_{date.today().isoformat()}.csv")
    print(output)
    # wfs = low_wfs + mid_wfs
    params = set(zip(wfs, [queue for x in range(len(wfs))]))
    # for thruple in params:
    #     print(thruple)  # help determine which order should be in file
    with Pool(processes=6) as pool:
        lstn = pool.apply_async(listener, (queue, output))
        result = pool.starmap(run_parametric, params)
    with Pool(processes=6) as pool:
        lstn = pool.apply_async(listener, (queue, output))
        result = pool.starmap(run_shadow, params)

        queue.put('Finish')
