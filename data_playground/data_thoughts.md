# Data output from simulations

Standard config for the TopSim simulator is to store the relative paths for observation workflows in in the 'pipelines' sub-dictionary of the 'instrument' dictionary. This saves having to store full file paths, but also makes it harder to 'find' the old workflow files if the config is being used on a new machine. 
 
 What we can do when storing the workflow files for each observation in HDF5 is create a 'path' that reflects the original using the path-like key storage that exists in HDF5.
 
 In fact, we might as well store all the files, and just separate based on the config directory and scrub out the `.json` for the file paths. Then we can store original file paths and complete config data in tables. 
 
 
 The current approach is to: 
    - Store files on the 
 
 
 