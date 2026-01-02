## Test Modlow script

setup conda environment through: 

    bash env.sh create

then run the test `proxy_model.py` by: 

    bash env.sh run python proxy_model.py
    
      
    
## Using Dask to implement distributed map

TODO: test and update with conda environmnet
    
or test distributed calculation by:

    bash env.sh  pbs_run.sh
    
features:
- dask scheduler called directly from the starting process
- dask worker called through 'simple_dsh'
  own ssh based implementation of a simplified distributed shell (dsh) 
- automatic wait of workers until the scheduler is ready
- automatic teardown through the trap mechanism
- payload python file: 'run_map.py'
  demonstrates 'run_in_subprocess' decorator to execute a function in a subprocess
