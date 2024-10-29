# Copyright (C) 9/5/22 RW Bunney
import json
import random

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
import logging
import time
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
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
PAR_MODEL_SIZING = Path(
    "/home/rwb/github/skaworkflows/skaworkflows/data/sdp-par-model_output/.archive/2021"
    "-06-02_long_HPSOs.csv"
)

TESTING = False

def run_shadow(params: dict, tup: tuple):
    output, lock = tup
    env = Environment(params["cfg"], dictionary=True)
    wf_path = Path(params["dir"]) / params["workflow"]
    workflow = Workflow(wf_path)
    workflow.add_environment(env)
    LOGGER.info(
        "Running FCFS for observation %s using %s",
        params['observation'], params['workflow']
    )
    params['method'] = 'heft'
    params['time'] = heft(workflow).makespan
    params["graph_type"] = ".".join(params["graph_type"])
    # # heft_res = None
    output_params = {k: i for k, i in params.items() if k != 'cfg'}
    for k, i in output_params.items():
        LOGGER.debug("Param: %s, Value: %s", k, i)
    LOGGER.debug("Graph type: %s", output_params["graph_type"])
    res_str_fcfs = ','.join([str(x) for x in output_params.values()])
    LOGGER.info("FCFS result: %s", res_str_fcfs)

    if not TESTING:
        lock.acquire()
        try:
            with output.open('a') as f:
                # time.sleep(1)
                f.write(f"{res_str_fcfs}\n")
                f.flush()
        finally:
            lock.release()
        return


def run_parametric(params: dict, tup: tuple, test=False):
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
    output, lock = tup

    LOGGER.info(
        "Running Parametrics for observation %s using %s",
        params['observation'], params['workflow']
    )
    params['method'] = 'heft'

    # Extract observation from params and fetch the SDP Parametric model runtime estimates
    # For maximal use case
    observation = params['observation']
    result = calculate_parametric_runtime_estimates(
        PAR_MODEL_SIZING, params['telescope'], [observation], params['graph_type']
    )
    duration = result[observation]["total_flops"] / (result[observation]["batch_flops"]),
    params['time'] = duration

    # Ensure workflows is not separated by comma, so as to avoid CSV compatibility issues
    params["graph_type"] = ".".join(params["graph_type"])

    # # heft_res = None
    output_params = {k: i for k, i in params.items() if k != 'cfg'}
    for k, i in output_params.items():
        LOGGER.debug("Param: %s, Value: %s", k, i)

    LOGGER.debug("Graph type: %s", output_params["graph_type"])
    res_str_fcfs = ','.join([str(x) for x in output_params.values()])
    LOGGER.info("FCFS result: %s", res_str_fcfs)
    if not TESTING:
        lock.acquire()
        try:
            with output.open('a') as f:
                # time.sleep(1)
                f.write(f"{res_str_fcfs}\n")
                f.flush()
        finally:
            lock.release()
        return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(Path(__file__).name,)
    parser.add_argument('path')
    parser.add_argument('-v', '--verbose')

    args = parser.parse_args()
    BASE_DIR = Path(args.path)
    if not BASE_DIR.exists():
        logging.warning("Path %s does not exist, leaving script...", str(BASE_DIR))
    # TODO check if path is file path or directory; if file path, flag this, get
    # base_dir and then skip the loop once using the file path given to the script
    try:
        level = args.verbose
        LOGGER.setLevel(level=logging.DEBUG)
    except IndexError:
        LOGGER.warning("Could not set log level, using default 'Warning'. ")

    params = []
    shadow_config = {}
    total_config = 0
    for cfg_path in os.listdir(BASE_DIR):
        if (BASE_DIR / cfg_path).is_dir():
            continue
        print(BASE_DIR / cfg_path)
        total_config += 1
        # Setup for SHADOW config
        shadow_config = config_to_shadow(BASE_DIR / cfg_path)

        # Retrieve workflow parameters

        # TODO consider adding this to SKAWorkflows library

        with open(BASE_DIR / cfg_path) as fp:
            cfg = json.load(fp)
        telescope_type = cfg["instrument"]["telescope"]["observatory"]
        pipelines = cfg["instrument"]["telescope"]["pipelines"]
        nodes = len(cfg["cluster"]["system"]["resources"])
        observations = pipelines.keys()
        LOGGER.debug('Observations: %s', observations)
        parameters = (
            pd.DataFrame.from_dict(pipelines, orient="index")
            .reset_index()
            .rename(columns={"index": "observation"})
        )
        parameters["nodes"] = nodes
        parameters["dir"] = BASE_DIR
        # Append information necessary for paramteric runner
        parameters["telescope"] = telescope_type + '-adjusted'

        for i in range(len(observations)):
            observation = dict(parameters.iloc[i])
            # Observations stored in TOpSim format have _N appended to the end.
            # We do not need this for the scheduling tests
            observation['observation'] = observation['observation'].split('_')[0]
            params.append(observation)
        for o in params:
            o["cfg"] = shadow_config
        # break # We only care about a single config file.

    LOGGER.info("Total configs processed: %d", total_config)
    LOGGER.info("Number of observations added %d", len(params))
    for i, p in enumerate(params):
        p['time'] = i

    if not params:
        LOGGER.warning("No inputs provided, will not run scheduling")
        sys.exit()

    header = ','.join([p for p in params[0].keys() if p != 'cfg'])

    manager = Manager()
    queue = manager.Queue()
    lock = manager.Lock()
    LOGGER.debug("Writing .csv with header: %s ", header)
    output = Path(__file__).parent / f"results_{date.today().isoformat()}.csv"
    with output.open('w+') as fo:
        fo.write(f"{header}\n")
        fo.flush()
        from itertools import product

        with Pool(processes=1) as pool:
            pool.starmap(run_parametric, product(params, [(output, lock)]))
        with Pool(processes=3) as pool:
            pool.starmap(run_shadow, product(params, [(output, lock)]))
