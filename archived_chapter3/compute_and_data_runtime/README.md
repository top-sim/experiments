# Investigating dynamic range of workflow applications

## Overview

When we perform the workflow characterisation based on the parametric model, 
we are segregating the workflow tasks (such as Gridding, Visibility Subtraction etc.) across hundreds of nodes in the graph. This is coarsely/simply done, so each 'Gridding' task has the same cost. 

When we allocate a task to a computing nodes, we can calculate the runtime of that task on that node (a simple division). 