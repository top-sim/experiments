# Copyright (C) 15/9/21 RW Bunney

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

"""
Experimentation with pandas and HDF5

Intention moving forward is to improve the granularity of the output of our
simulations, and to store original configuration data in the event that
original files are lost or relative paths are misleading.
"""

import os
import json
import logging
import datetime
import pandas as pd

logging.basicConfig(level='DEBUG')
LOGGER = logging.getLogger(__name__)


def stringify_json_data(path):
    """
    From a given file pointer, get a string representation of the data stored

    Parameters
    ----------
    fp : file pointer for the opened JSON file

    Returns
    -------
    jstr : String representation of JSON-encoded data

    Raises:

    """

    try:
        with open(path) as fp:
            jdict = json.load(fp)
    except json.JSONDecodeError:
        raise

    jstr = json.dumps(jdict)  # , indent=2)
    return jstr


def create_config_table(path):
    """
    From the simulation config files, find the paths for each observation
    workflow and produce a table of this information

    Parameters
    ----------
    path

    Returns
    -------

    """
    cfg_str = stringify_json_data(path)
    jdict = json.loads(cfg_str)
    pipelines = jdict['instrument']['telescope']['pipelines']
    ds = [['simulation_config', path, cfg_str]]
    for observation in pipelines:
        p = pipelines[observation]['workflow']
        p = p.replace('publications', 'archived_results')
        wf_str = stringify_json_data(p)
        tpl = [f'{observation}', p, wf_str]
        ds.append(tpl)

    df = pd.DataFrame(ds, columns=['entity', 'config_path', 'config_json'])
    return df


def reverse_engineer_hdf_config(hdf5_key):
    """
    Given the hdf5 key, convert string-representations of JSON config to the
    original configuration files so it is possible to reproduce results.

    Parameters
    ----------
    hdf5_key

    Returns
    -------

    """


def compile_separate_experiments_pkl_to_hdf5(
        simpath, taskpath, outpath, timestamp, column='config'
):
    """
    Given a path of experiments, grab the results and produce hdf5 instances to
    bring everything together for the purposes of subsequent analysis.
    Parameters
    ----------
    path : str
        path to the pickle files
    columns : List
        list of columns that we are differentiating.

    Returns
    -------

    """

    if os.path.exists(outpath):
        LOGGER.warning('Output HDF5 path already exists, appending timestamp')
        ts = datetime.datetime.today().strftime("%y-%M-%d-%H-%M-%S")
        tmp_path = f'{outpath}_{ts}'
        store = pd.HDFStore(tmp_path)
    else:
        store = pd.HDFStore(outpath)

    if not os.path.exists(simpath):
        raise FileNotFoundError(simpath)
    elif not os.path.exists(taskpath):
        raise FileNotFoundError

    dfsim = pd.read_pickle(simpath)
    dftask = pd.read_pickle(simpath)
    # Historically, we've differentiated by config file names, so this is the
    # default parameter for 'column', on which we will split the dataframe
    different_sim_config = set(dfsim[column])
    # algorithm = set(dfsim['algorithm'])
    for sim in different_sim_config:
        tmp_sim_df = dfsim[dfsim[column] == sim]
        tmp_task_df = dftask[dftask[column] == sim]
        config_path = sim.replace('publications', 'archived_results')
        if 'archived_results' not in sim:
            config_path = f'archived_results/{sim}'

        workflows = create_config_table(config_path)
        sanitised_path = config_path.strip('.json').split('/')[-1]

        store.put(key=f'{timestamp}/{sanitised_path}/sim', value=tmp_sim_df)
        store.put(key=f'{timestamp}/{sanitised_path}/tasks', value=tmp_task_df)
        store.put(key=f'{timestamp}/{sanitised_path}/config', value=workflows)

    store.close()

    return outpath


if __name__ == '__main__':
    # monolithic_plottiang_code()
    DATA_DIR = 'visualisation_playground/playground_data'
    SIM_PATH = f'{DATA_DIR}/2021-07-08_global_sim_greedy_from_plan.pkl'
    TASK_PATH = f'{DATA_DIR}/2021-07-08_global_tasks_greedy_from_plan.pkl'
    OUTPATH = f'{DATA_DIR}/2021-07-08-greedy.h5'
    timestamp = (datetime.datetime(2021, 7, 8).strftime("%y_%m_%d_%H_%M_%S"))
    compile_separate_experiments_pkl_to_hdf5(SIM_PATH, TASK_PATH, OUTPATH,
                                             timestamp=f'd{timestamp}')

    # TODO - Create a single data frame from one of the simulations
    # TODO - get a single workflow file from the simulation to test the
    #  strings are the same
    # TODO - simulation framework for data storage; when we create a
    #  simulation, do we use HDF5 as the individual data storage?
