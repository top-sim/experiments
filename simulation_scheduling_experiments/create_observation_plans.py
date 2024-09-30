import os
import json
import random
import sys
from pathlib import Path
import logging

import numpy as np

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
SKA_channels = [64, 128, 256,]
# SKA_low_channels = [64, 128, 256, ]

# 32 is an arbitrary minimum; 512 hard maximum
SKA_Low_antenna = [64, 128, 256, 512]
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
    # for channels in SKA_channels:
    #     for demand in SKA_Low_antenna:
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
                # Consider putting a hard limit on the max channels if
                # generating workflows for a 896 syste?
                "coarse_channels": 896,  # parallel channels,
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
    SKA_channels = [0]
    for channels in SKA_channels:
        for i in range(len(SKA_Low_antenna)):
            demand = SKA_Low_antenna[i]
            channels_demand = 128
            # if demand > 256:
            #     channels_demand = 256
            alt_demand = min(demand*2, SKA_Low_antenna[-1]) # was -1
            # alt_channels = min(channels*2, SKA_channels[-1])

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
                        "channels": channels_demand * channel_multiplier,
                        "coarse_channels": alt_demand,
                        "baseline": 65000.0,
                        "telescope": "low"
                    },
                    {
                        "count": 4,
                        "hpso": 'hpso02a',
                        "duration": low['hpso02a']['duration'],
                        "workflows": low['hpso02a']['workflows'],
                        "demand": demand,
                        "channels": channels_demand * channel_multiplier,
                        "coarse_channels": demand,
                        "baseline": 65000.0,
                        "telescope": "low"
                    },
                    {
                        "count": 4,
                        "hpso": 'hpso02b',
                        "duration": low['hpso02b']['duration'],
                        "demand": demand,
                        "workflows": low['hpso02b']['workflows'],
                        "channels": channels_demand * channel_multiplier,
                        "coarse_channels": demand,
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

def create_demand_ratio_permutations():
    x = [64,64,64]
    n=3
    max = 512
    print(len(x))
    random.seed(0)
    for j in [64, 128, 256, 512]:
        for i in range(0,len(x)):
            idx = random.randint(0, len(x)-1)
            x[idx] = j
            print(x)
            number_obs = np.array([1, 2, 2])*n
            print(number_obs, sum(number_obs))
            print(sum(np.array(x)*number_obs)/(sum(number_obs)*max))

def varied_low_obs_plan():
    params = []
    SKA_channels = [0]

    for channels in SKA_channels:
        for i in range(len(SKA_Low_antenna)):
            demand = SKA_Low_antenna[i]
            channels_demand = 128
            alt_demand = min(demand*2, SKA_Low_antenna[-1])
            # alt_channels = min(channels*2, SKA_channels[-1])

            observation = {
                "nodes": 896,
                "infrastructure": "parametric",
                "telescope": "low",
                "items": [
                    {
                        "count": random.randint(3,5) ,
                        "hpso": "hpso01",
                        "duration": low['hpso01']['duration'],
                        "workflows": low['hpso01']['workflows'],
                        "demand": alt_demand,
                        "channels": channels_demand * channel_multiplier,
                        "coarse_channels": demand,
                        "baseline": 65000.0,
                        "telescope": "low"
                    },
                    {
                        "count": random.randint(6,10) ,
                        "hpso": 'hpso02a',
                        "duration": low['hpso02a']['duration'],
                        "workflows": low['hpso02a']['workflows'],
                        "demand": demand,
                        "channels": channels_demand * channel_multiplier,
                        "coarse_channels": demand,
                        "baseline": 65000.0,
                        "telescope": "low"
                    },
                    {
                        "count": random.randint(6,10) ,
                        "hpso": 'hpso02b',
                        "duration": low['hpso02b']['duration'],
                        "demand": demand,
                        "workflows": low['hpso02b']['workflows'],
                        "channels": channels_demand * channel_multiplier,
                        "coarse_channels": demand,
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

all_params = []
if telescope == 'low':
    all_params.append(standard_low_obs_plan())
    all_params.append(varied_low_obs_plan())
elif telescope == 'mid':
    all_params.append(standard_mid_obs_plan())
else:
    logging.info("Creating simple obs plan")
    all_params.append(simple_single_obs_plan())

low_path = BASE_DIR / "chapter4/test_topsim_variations_demand" / telescope

# params = simple_single_obs_plan()
# HPSO_PLANS=os.listdir("chapter4/scalability/simfiles/mid")
print("Creating config")
for ap in all_params:
    for plan in ap:
        create_config(
            plan,
            low_path,
            WORKFLOW_TYPE_MAP,
            timestep=5,
            data=False,
            data_distribution='standard',
            multiple_plans=True)
        create_config(
            plan,
            low_path,
            WORKFLOW_TYPE_MAP,
            timestep=5,
            data=True,
            data_distribution='edges',
            multiple_plans=True)
