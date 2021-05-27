# 2021 ISC-HPC config files

This directory contains configuration files for the plots that were produced
 as preliminary results for a poster and video presentation at the 2021 ISC
 -HPC PhD Forum. The cluster is a 40-machine homogeneous cluster with the
  compute values generated randomly. 
 
 * `workflows/` contains basic prototype workflows that are modelled on the shape of an SKA continuum pipeline. These were generated using EAGLE, the DALiuGE logical translator, and an experimental pipeline generation that is currently in-development (see `github.com/top-sim/topsim_pipelines`). The compute values were generated at random; in the future, these values will be based on expected computational loads. 
 * `individual/` contains observations schedule involving a single
  observation/workflow. 
* `single_size/` contains observation schedules with multiple observations, some overlapping, all with the same sized workflow. This was intended to determine how the scheduling approaches dealt with overlapping workflows.  

Data used to create the plots used for ISC-HPC are from the `40cluster` directories. 
