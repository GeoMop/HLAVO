# Using Dask to implement distributed map

setup as: 

    bash setup_env

Only this step uses python system module to setup python venv.
    
run as: 

    bash pbs_run.sh

    
features:

- dask scheduler called directly from the starting process
- dask worker called through 'simple_dsh'
  own ssh based implementation of a simplified distributed shell (dsh) 
- automatic wait of workers until the scheduler is ready
- automatic teardown through the trap mechanism
- payload python file: 'run_map.py'
  demonstrates 'run_in_subprocess' decorator to execute a function in a subprocess
