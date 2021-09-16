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
import tables
import pandas as pd


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
        wf_str = stringify_json_data(p)
        tpl = [f'{observation}', p, wf_str]
        ds.append(tpl)

    df = pd.DataFrame(ds, columns=['entity', 'config_path', 'config_json'])
    return df


def multi_sim_config_tables(cfg):
    """
    For a simulation, gather the config files and turn them into strings to
    then place in a table
    Parameters
    ----------
    cfg

    Returns
    -------
    pandas.dataframe
    """


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

def compile_separate_experiments_pkl_to_hdf5(path):
    """
    Given a path of experiments, grab the results and produce hdf5 instances to
    bring everything together for the purposes of subsequent analysis.
    Parameters
    ----------
    path

    Returns
    -------

    """

if __name__ == '__main__':
    logging.basicConfig(level='DEBUG')
    LOGGER = logging.getLogger(__name__)

    # simulation for batch scheduling
    batch_pkl = 'data_playground/2021-09-07-sim.pkl'
    tasks_pkl = 'data_playground/2021-09-07-tasks.pkl'
    sim_config = 'data_playground/mos_sw80.json'
    workflow_config = 'data_playground/shadow_Continuum_ChannelSplit_80.json'
    HDF5_PATH = 'data_playground/experiment.h5'
    PKL_TO_HDF_PATH = 'data_playground/2021-09-07-sim.h5'

    if os.path.exists(HDF5_PATH):
        os.remove(HDF5_PATH)

    cfg_str = stringify_json_data(sim_config)
    LOGGER.debug(f'cfg_str: {cfg_str}')

    wf_str = stringify_json_data(workflow_config)
    LOGGER.debug(f'wf_str:{wf_str}')

    if os.path.exists(PKL_TO_HDF_PATH):
        os.remove(PKL_TO_HDF_PATH)
        LOGGER.debug(f"Creating: {PKL_TO_HDF_PATH}...")
    batch_df = pd.read_pickle(batch_pkl)
    batch_df.to_hdf(PKL_TO_HDF_PATH, key='df')
    tasks_df = pd.read_pickle(tasks_pkl)
    store = pd.HDFStore(HDF5_PATH)
    LOGGER.debug(store.info())

    # batch_hdf = tables.open_file(PKL_TO_HDF_PATH, 'w')
    # LOGGER.debug(batch_hdf)

    # with open(workflow_config) as fp:
    sample_df = pd.DataFrame([[workflow_config, wf_str]],
                             columns=['path', 'data'])
    sanitised_path = workflow_config.strip('.json')
    # store.put(key=f'wf_cfg/{sanitised_path}', value=sample_df)
    LOGGER.debug(store.info())
    key = f'sim_results/{sanitised_path}'

    # with open('experiment.json', 'w') as fp:
    #     jdict = json.loads(store[f'wf_cfg/{sanitised_path}'].iloc[0]['data'])
    #     json.dump(jdict,fp, indent=2)

    sanitised_path = os.path.basename(sim_config).strip('.json')
    store.put(key=f'{sanitised_path}/global', value=batch_df)

    store.put(key=f'{sanitised_path}/tasks', value=tasks_df)
    # LOGGER.debug(store.info())

    workflows = create_config_table(sim_config)
    # LOGGER.debug(workflows)
    store.put(key=f'{sanitised_path}/config',
              value=workflows)

    LOGGER.debug(f"Data\n{store.info()}")
    store.close()
