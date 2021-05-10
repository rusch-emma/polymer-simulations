#!/usr/bin/env python

#SBATCH --nodes=1
#SBATCH --time=08:00:00 
#SBATCH --partition=g 
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-cpu=8G
#SBATCH --qos=g_short
#SBATCH --constraint=g3

import sys
import yaml
import subprocess
import os
import numpy as np
import itertools
import logging

from pathlib import Path

"""
Runs a sweep over all provided parameter combinations for a specified Polychrom simulation script. For an example see /config/example_config.yml.

Provide a .yml configuration file with the following fields:
    app: str
        Path to the Python script which will run the simulation.
    out_dir: str
        Output directory for the results.
    replicates: int
        The number of replicates to run for each parameter combination.
    cli_params: dict
        A dictionary of all CLI options the simulation script accepts with keys as the option names and values as lists of all possible option values.
        Simulations for each combination of parameters will be run.

    Optionally (useful for downstream analyses tools):
    plots_dir: str
        Directory in which to store plot files.
    maps_dir: str
        Directory in which to store contact map data in .npy format.
"""

logging.basicConfig(level=logging.INFO)

dry_run = 'SLURM_ARRAY_TASK_ID' not in os.environ
if dry_run:
    logging.warning('SLURM_ARRAY_TASK_ID not found, executing a dry run!')
task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID',0))
logging.info(f'Executing task # {task_id}')

cfg_file = sys.argv[1]
with open(cfg_file) as yml_file:
    config = yaml.load(yml_file, Loader=yaml.FullLoader)

run_file = config['app']
out_dir = config['out_dir']

cli_params = config['cli_params']
cli_params['--replicate'] = list(range(config['replicates']))
cli_params['--id'] = [task_id]

N_PARAM_COMBOS = np.prod([len(v) for v in cli_params.values()])
logging.info(f'Sweeping over {N_PARAM_COMBOS} parameter combinations!')


cli_param_set = next(itertools.islice(
                        itertools.product(*[v for v in cli_params.values()]),
                        task_id, task_id+1)
                    )
    
logging.info(f'Param set 1: {cli_param_set}')

cli_param_set = {k:v for k,v in zip(cli_params.keys(), cli_param_set)}

logging.info(f'Param set 2: {cli_param_set}')

# store replicates of parameter combinations in their own directories
id = '-'.join([k[2:] + str(v) for k, v in cli_param_set.items() if k != '--replicate' and k != '--id'])
out_folder = os.path.join(out_dir, id)
Path(out_folder).mkdir(exist_ok=True)

cmd = [
    'python',
    run_file,
    '--out_folder',
    out_folder
]

for name, value in cli_param_set.items():
    if value is not None:
        cmd.append(name)
        if bool(str(value)):
            cmd.append(str(value))

logging.info(f'Executing command: {repr(cmd)}' )

if not dry_run:
    subprocess.call(cmd)
