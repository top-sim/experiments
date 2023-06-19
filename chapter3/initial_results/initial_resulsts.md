# Results were generated with the following parameters.

# Data generated with generate_data.py
- Maximam imaging workflows (ICAL, DPrepA-DPrepD)
- Workflows with and without data 
- Data on tasks and edges 
- Channels: 512 and 896
- Compute nodes: 896 only 
- Parametric model: Also '896' node equivalent; this means not reserving nodes for ingest, and scheduling across the entire cluster

# Experiments run using run_comparisons.py
- Used FCFS from SHADOW scheduling library 
- Parametric model calculate_parametric_runtime_estimates from `skaworkflows`
- Incomplete formatting of data (no heading row) 
- Duplicates in the run 