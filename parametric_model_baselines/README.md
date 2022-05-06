# Experiment summary

This experiment aims to demonstrate how our workflow-based scheduling model differs from the `sdp-par-model`, specifically with respect to how offline/batch-processing is calculated. 

## Scripts

* `generate_data.py` produces workflows based on selected HPSOs in specified sub-directories
    * This will use 
* `sanity_check.py` double-checks that each workflow's compute costs are correct re: `sdp-par-model` (using `skaworkflows.workflow_analysis`) 
* `run_shadow.py` iterates through each directory and schedules accordingly
* `compare.py` produces plots that map the `skaworkflows.parametric_runner` results to the `run_shadow.py` results
 