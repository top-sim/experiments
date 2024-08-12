import os
import json
import sys
from pathlib import Path
import logging

logging.basicConfig(level = logging.INFO)

from skaworkflows.config_generator import create_config

low = {'hpso01': {"duration": 18000,
                  'workflows': ["ICAL", "DPrepA", "DPrepB"]},  # , "DPrepC, DPrepD"]},
       'hpso02a': {"duration": 18000,
                   'workflows': ["ICAL", "DPrepA", "DPrepB"]},  # }, "DPrepC, DPrepD"]},
       'hpso02b': {"duration": 18000,
                   'workflows': ["ICAL", "DPrepA", "DPrepB"]}}  # , "DPrepC, DPrepD"]}}

# low = {'hpso01': {"duration": 18000,
#                   'workflows': ["ICAL"]},
#        # , "DPrepA", "DPrepB"]},  # , "DPrepC, DPrepD"]},
#        'hpso02a': {"duration": 18000,
#                    'workflows': ["ICAL"]},
#        # , "DPrepA", "DPrepB"]},  # }, "DPrepC, DPrepD"]},
#        'hpso02b': {"duration": 18000,
#                    'workflows': [
#                        "ICAL"]}}  # , "DPrepA", "DPrepB"]}}  # , "DPrepC, DPrepD"]}}
mid = {
    'hpso13': {'duration': 28800, 'workflows': ["ICAL", "DPrepA", "DPrepB", "DPrepC"]},
    'hpso15': {'duration': 15840, 'workflows': ["ICAL", "DPrepA", "DPrepB", "DPrepC"]},
    'hpso22': {'duration': 28800, 'workflows': ["ICAL", "DPrepA", "DPrepB"]},
    'hpso32': {'duration': 7920, 'workflows': ["ICAL", "DPrepB"]}
}
#
# These are the 'coarse-grained' channel values.
SKA_channels = [128,] #  256, 512]
SKA_low_channels = [64, 128, 256, ]

# 32 is an arbitrary minimum; 512 hard maximum
SKA_Low_antenna = [32, 64, 128, 256, ]
# 64 + N, where 64 is the base number of dishes
SKA_Mid_antenna = [64, 102, 140, 197]

channel_multiplier = 128

def simple_single_obs_plan():
    """
    Generate a plan for a single observation

    This is purely to validate and demonstrate the scheduling heuristic performance.

    Returns
    -------
    plans: list of json-compliant dictionaries, representing observation plans
    """
    params = []
    for channels in SKA_channels:
        for demand in SKA_Low_antenna:
            observation = {
                "nodes": 896,
                "infrastructure": "parametric",
                "telescope": "low",
                "items": [
                    {
                        "count": 1,
                        "hpso": "hpso01",
                        "duration": low['hpso01']['duration'],
                        "workflows": low['hpso01']['workflows'],
                        "demand": 512, #demand * 1,
                        "channels": 256*channel_multiplier,# channels * channel_multiplier,
                        "coarse_channels": 512, #896,  # channels,
                        "baseline": 65000.0,
                        "telescope": "low"
                    }
                ]
            }
            params.append(observation)

    return params


def standard_mid_obs_plan(verbose=False):
    """
    Currently, this is a placeholder method to generate one of a couple different
    observation plans.

    Expect this method to be a) renamed in the future and b) improved upon

    'hpso13': {'duration': 28800, 'workflows': ["ICAL", "DPrepA", "DPrepB", "DPrepC"]},
    'hpso15': {'duration': 15840, 'workflows': ["ICAL", "DPrepA", "DPrepB", "DPrepC"]},
    'hpso22': {'duration': 28800, 'workflows': ["ICAL", "DPrepA", "DPrepB"]},
    'hpso22': {'duration': 28800, 'workflows': ["ICAL", "DPrepA", "DPrepB"]},
    'hpso32': {'duration': 7920, 'workflows': ["ICAL", "DPrepB"]}


    Returns
    -------

    """

    count = 0
    params = []
    for channels in SKA_channels:
        for i in range(len(SKA_Mid_antenna)-1):
            demand = SKA_Mid_antenna[i]
            alt_demand = SKA_Mid_antenna[i+1]
            observation = {
                "nodes": 512,
                "infrastructure": "parametric",
                "telescope": "mid",
                "items": [
                    {
                        "count": 1,
                        "hpso": "hpso13",
                        "duration": mid['hpso13']['duration'],
                        "workflows": mid['hpso13']['workflows'],
                        "demand": demand,
                        "channels": channels * channel_multiplier,
                        "coarse_channels": channels,
                        "baseline": 65000.0,
                        "telescope": "low"
                    },
                    {
                        "count": 2,
                        "hpso": 'hpso15',
                        "duration": mid['hpso15']['duration'],
                        "workflows": mid['hpso15']['workflows'],
                        "demand": alt_demand,
                        "channels": channels * channel_multiplier,
                        "coarse_channels": channels,
                        "baseline": 65000.0,
                        "telescope": "low"
                    },
                    {
                        "count": 2,
                        "hpso": 'hpso22',
                        "duration": mid['hpso22']['duration'],
                        "demand": demand,
                        "workflows": mid['hpso22']['workflows'],
                        "channels": channels * channel_multiplier,
                        "coarse_channels": channels,
                        "baseline": 65000.0,
                        "telescope": "low"
                    },
                    {
                        "count": 4,
                        "hpso": 'hpso32',
                        "duration": mid['hpso32']['duration'],
                        "demand": demand,
                        "workflows": mid['hpso32']['workflows'],
                        "channels": channels * channel_multiplier,
                        "coarse_channels": channels,
                        "baseline": 65000.0,
                        "telescope": "low"
                    }
                ]
            }

            params.append(observation)
            # path_name = dir_name / 'low' / f"low_{count}_.json"
            # with path_name.open('w') as fp:
            #     json.dump(observation, fp, indent=2)
    if verbose:
        print(json.dumps(params, indent=2))
    return params


def standard_low_obs_plan():
    """
    Currently, this is a placeholder method to generate one of a couple different
    observation plans.

    Expect this method to be a) renamed in the future and b) improved upon

    Returns
    -------

    """

    count = 0
    params = []
    for channels in SKA_channels:
        for i in range(len(SKA_Low_antenna)):
            demand = SKA_Low_antenna[i]
            alt_demand = SKA_Low_antenna[-1]

            observation = {
                "nodes": 896,
                "infrastructure": "parametric",
                "telescope": "low",
                "items": [
                    {
                        "count": 2,
                        "hpso": "hpso01",
                        "duration": low['hpso01']['duration'],
                        "workflows": low['hpso01']['workflows'],
                        "demand": alt_demand,
                        "channels": channels * channel_multiplier,
                        "coarse_channels": channels,
                        "baseline": 65000.0,
                        "telescope": "low"
                    },
                    {
                        "count": 4,
                        "hpso": 'hpso02a',
                        "duration": low['hpso02a']['duration'],
                        "workflows": low['hpso02a']['workflows'],
                        "demand": demand,
                        "channels": channels * channel_multiplier,
                        "coarse_channels": channels,
                        "baseline": 65000.0,
                        "telescope": "low"
                    },
                    {
                        "count": 4,
                        "hpso": 'hpso02b',
                        "duration": low['hpso02b']['duration'],
                        "demand": demand,
                        "workflows": low['hpso02b']['workflows'],
                        "channels": channels * channel_multiplier,
                        "coarse_channels": channels,
                        "baseline": 65000.0,
                        "telescope": "low"
                    }
                ]
            }
            params.append(observation)
            # path_name = dir_name / 'low' / f"low_{count}_.json"
            # with path_name.open('w') as fp:
            #     json.dump(observation, fp, indent=2)
    return params


graph_type = sys.argv[1]
telescope = sys.argv[2]

WORKFLOW_TYPE_MAP = {
    "ICAL": graph_type,
    "DPrepA": graph_type,
    "DPrepB": graph_type,
    "DPrepC": graph_type,
    "DPrepD": graph_type,
}

BASE_DIR = Path("/home/rwb/Dropbox/University/PhD/experiment_data/")

if telescope == 'low':
    params = standard_low_obs_plan()
elif telescope == 'mid':
    params = standard_mid_obs_plan()
else:
    logging.info("Creating simple obs plan")
    params = simple_single_obs_plan()

low_path = BASE_DIR / "chapter4/test_topsim" / telescope

# params = simple_single_obs_plan()
# HPSO_PLANS=os.listdir("chapter4/scalability/simfiles/mid")
print("Creating config")
for plan in params:
    create_config(
        plan,
        low_path,
        WORKFLOW_TYPE_MAP,
        timestep=5,
        data=False,
        data_distribution='standard',
        multiple_plans=False)
    create_config(
        plan,
        low_path,
        WORKFLOW_TYPE_MAP,
        timestep=5,
        data=True,
        data_distribution='edges',
        multiple_plans=False)
