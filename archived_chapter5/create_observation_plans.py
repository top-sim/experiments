import os
import json
import sys
from pathlib import Path

low = {'hpso01': {"duration": 18000,
                  'workflows': ["ICAL", "DPrepA", "DPrepB"]},  # , "DPrepC, DPrepD"]},
       'hpso02a': {"duration": 18000,
                   'workflows': ["ICAL", "DPrepA", "DPrepB"]},  # }, "DPrepC, DPrepD"]},
       'hpso02b': {"duration": 18000,
                   'workflows': ["ICAL", "DPrepA", "DPrepB"]}}  # , "DPrepC, DPrepD"]}}
# mid_hpsos = {
#     'hpso13': {'duration': 28800, 'workflows': ["ICAL", "DPrepA", "DPrepB", "DPrepC"]},
#     'hpso15': {'duration': 15840, 'workflows': ["ICAL", "DPrepA", "DPrepB", "DPrepC"]},
#     'hpso22': {'duration': 28800, 'workflows': ["ICAL", "DPrepA", "DPrepB"]},
#     'hpso32': {'duration': 7920, 'workflows': ["ICAL", "DPrepB"]}
# }

# These are the 'coarse-grained' channel values.
SKA_channels = [128] # [64, 128, 256]

# 32 is an arbitrary minimum; 512 hard maximum
# SKA_Low_antenna = [32, 64, 128, 256, ]
SKA_Low_antenna = [64]  # 128]
# 64 + N, where 64 is the base number of dishes
SKA_Mid_antenna = [64, 102, 140, 197]

channel_multiplier = 128
# dir_name = Path(__file__).parent / 'simfiles'
# if dir_name.exists():
#     print("Need to delete files first.")
#     exit(1)
# dir_name.mkdir()
# (dir_name / 'low').mkdir()
# (dir_name / 'mid').mkdir()

count = 0
params = []
for channels in SKA_channels:
    for demand in SKA_Low_antenna:
        observation = {
            "nodes": 512,
            "infrastructure": "parametric",
            "telescope": "low",
            "items": [
                {
                    "count": 2,
                    "hpso": "hpso01",
                    "duration": low['hpso01']['duration'],
                    "workflows": low['hpso01']['workflows'],
                    "demand": demand * 4,
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
        count += 1

from skaworkflows.config_generator import create_config

graph_type = sys.argv[1]

WORKFLOW_TYPE_MAP = {
    "ICAL": graph_type,
    "DPrepA": graph_type,
    "DPrepB": graph_type,
    "DPrepC": graph_type,
    "DPrepD": graph_type,
}
BASE_DIR = Path("/home/rwb/Dropbox/University/PhD/experiment_data/")

low_path = BASE_DIR / "chapter5/interdependency" / "schedule_experimentation"

# HPSO_PLANS=os.listdir("chapter4/scalability/simfiles/mid")
for plan in params:
    create_config(plan, low_path, WORKFLOW_TYPE_MAP, timestep=10, multiple_plans=True)
