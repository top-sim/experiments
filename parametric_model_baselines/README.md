# Experiment summary

## Todo: 

- [] Update mid-adjusted generated data to use 786 channels/compute nodes
- [] Update the sanity checks and run_comparisons for this too, as it should improve schedule time. 

This experiment aims to demonstrate how our workflow-based scheduling model differs from the `sdp-par-model`, specifically with respect to how offline/batch-processing is calculated. 

## Scripts

* `generate_data.py` produces workflows based on selected HPSOs in specified sub-directories
    * This will use 
* `sanity_check.py` double-checks that each workflow's compute costs are correct re: `sdp-par-model` (using `skaworkflows.workflow_analysis`) 
* `run_shadow.py` iterates through each directory and schedules accordingly
* `compare.py` produces plots that map the `skaworkflows.parametric_runner` results to the `run_shadow.py` results
 
 ## Graph terminology
 
 * `base` refers to the prototype continuum imaging pipeline that we are using as the basis for the workflows. This reflects the approximate structure of an SKA workflow. 
 * `parallel` refers to the parametric-equivalent graph. This is a parallel scatter across all nodes, and then completely sequential for each 'line' of the scatter. 
 
 ## Results (No data on nodes) 
 
 ### 2022_05_10_output.csv
 
 This file contains complete scatter of 896; that is, the channel parameters for the 'scatter' component of both graphs is 896, which is equivalent to the maximum number of nodes avaible on the low infrastructure. This leads to the parametric model, prototype workflow, and equivalent 'parallel/sequential' parametric workflow all showing the same time. 
 
 ### 2022_05_11_512channels_output.csv
 This file was generated after also generating new files ( by editing the generate_data.py script). In this example, we have limited the scatter to 512 to demonstrate how a 'more typical' workflow will look like on the system (for example, 512 is the maximum coarse grained channel binning the SKA will be working with).  It shows how reducing the frequency scatter increases the runtime of the workflow (and therefore a realistic model will not be able to reach the parametric model claims). 
 
 ### 2022_05_31_output_512nodes.csv
 This file is generated (after generating a new config.json) to demonstrate that the base-graph performance (which was better than the parallel-graph PM graph in the previous data due to 'machine stealing') is reduced to an equivalent makespan by restricting the number of nodes. 
 
 