# Copyright (C) 9/5/22 RW Bunney
import json

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
import os
from datetime import date
from copy import deepcopy
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, Manager
from shadow.models.workflow import Workflow
from shadow.models.environment import Environment
from shadow.algorithms.heuristic import heft, fcfs

sys.path.append("/home/rwb/github/skaworkflows")
from skaworkflows.config_generator import config_to_shadow
from skaworkflows.parametric_runner import calculate_parametric_runtime_estimates


# from chapter5.parametric_model_baselines.generate_data import (
#     LOW_HPSO_PATHS, MID_HPSO_PATHS)

PAR_MODEL_SIZING = Path("../sdp-par-model/2021-06-02_LongBaseline_HPSOs.csv")


def run_shadow(params, test=False):
    if params["data_distribution"] == "edges":
        return None
    #
    env = Environment(params["cfg"], dictionary=True)
    wf_path = Path(params["dir"]) / params["workflow"]
    workflow = Workflow(wf_path)
    workflow.add_environment(env)
    print(
        f"Running FCFS for observation {params['observation']} using {params['workflow']}"
    )
    fcfs_res = fcfs(workflow).makespan
    # # heft_res = None
    res_str_fcfs = (
        f"{params['observation']},"
        f"{params['workflow']},"
        f"{fcfs_res},"
        # f"{graph}," TODO Add graph type to output once this is complete
        f"{params['nodes']},"
        f"{params['demand']},"
        f"{params['channels']},"
        f"{params['data']},"
        f"{params['data_distribution']}"
    )
    print(res_str_fcfs)
    # lock.acquire()
    # with output.open('a') as f:
    #     f.write(f"{str(res_str_fcfs)}\n")
    #     f.flush()
    # lock.release()


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
    if graph == "scatter" or channels == 512 or "no_data" not in wf.name:
        return None
    if "no_data" in wf.name:
        data = False
    else:
        data = True
    hpso = f.split("_")[0]
    pipeline_list = None
    if pipeline_sets:
        pipeline_list = pipeline_sets.split("_")
    par_res = calculate_parametric_runtime_estimates(
        PAR_MODEL_SIZING, telescope, [hpso], pipeline_list
    )
    res_str = (
        f"{hpso},parametric,{par_res[hpso]['time']},"
        f"{graph},{telescope},{nodes},{channels},{data}"
    )
    print(res_str)
    lock.acquire()
    with output.open("a") as f:
        f.write(f"{str(res_str)}\n")
        f.flush()
    lock.release()


if __name__ == "__main__":
    # Given a simulation configuration file generated from a HPSO spec,
    # this script will: - Open the script and retrieve parameters for each
    # workflow - Retrieve the system sizing paramters (incl. number of
    # machines for the simulation) - Setup simulation combinations -
    # Translate simulation configuration to SHADOW format - Run combinations
    # using SHADOW library in parallel

    BASE_DIR = Path(
        "/home/rwb/Dropbox/University/PhD/experiment_data/chapter3"
        "/initial_results_newskaworkflows/low_maximal/prototype/"
    )
    params = []
    shadow_config = {}
    for cfg_path in os.listdir(BASE_DIR):
        if (BASE_DIR / cfg_path).is_dir():
            continue
        print(BASE_DIR / cfg_path)
        # Setup for SHADOW config
        shadow_config = config_to_shadow(BASE_DIR / cfg_path)

        # Retrieve workflow parameters
        with open(BASE_DIR / cfg_path) as fp:
            cfg = json.load(fp)
        telescope_type = cfg["instrument"]["telescope"]["observatory"]
        pipelines = cfg["instrument"]["telescope"]["pipelines"]
        nodes = len(cfg["cluster"]["system"]["resources"])
        observations = pipelines.keys()
        parameters = (
            pd.DataFrame.from_dict(pipelines, orient="index")
            .reset_index()
            .rename(columns={"index": "observation"})
        )
        parameters["nodes"] = nodes
        parameters["dir"] = BASE_DIR
        # print(parameters)
        for i in range(len(observations)):
            params.append(dict(parameters.iloc[i]))
        for o in params:
            o["cfg"] = shadow_config

        with Pool(processes=3) as pool:
            result = pool.map(run_shadow, params)
