#!/usr/bin/env python
"""Initialize the project's data space.

Iterates over all defined state points and initializes
the associated job workspace directories."""
import logging
import argparse
import signac
import numpy as np

def main(args):
    project = signac.init_project("cd")
    statepoints_init = []
    cases = ['neat_emimtfsi','acn_emimtfsi','acn_litfsi', 'wat_litfsi']
    # voltages = np.arange(0,3+0.25,0.25)
    voltages = [0, 1, 2, 3]
    seeds = [0, 1, 2, 3]
    for case in cases:
        for voltage in voltages:
            for seed in seeds:
                statepoint = dict(
                    case = case,
                    voltage = voltage,
                    seed = seed
                )
                project.open_job(statepoint).init()
                statepoints_init.append(statepoint) 
    
    # Writing statepoints to hash table as a backup
    project.write_statepoints(statepoints_init)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize the data space.")
    parser.add_argument(
        "-n",
        "--num-replicas",
        type=int,
        default=1,
        help="Initialize multiple replications.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
    