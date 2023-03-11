import os
import signac
import numpy as np
project = signac.get_project()

def copy_file(file_name, job_id, v, case, seed):
    for job in project.find_jobs({'voltage': 0, 'case': case, 'seed': seed}):
        rahman_job_id = job
        
    command_line = 'rsync -r workspace/{}/{} workspace/{}/'.format(rahman_job_id, file_name, job_id)
    print(command_line)
    os.system(command_line)
    
# voltages = np.arange(0,3+0.25,0.25)
# voltages = [0, 1, 2, 3]
voltages = [1,2,3]
cases = ['neat_emimtfsi', 'acn_emimtfsi', 'acn_litfsi', 'wat_litfsi']
seeds = [0,1,2,3]
i = 0
for seed in seeds:
    for case in cases:
        for v in voltages:
            for job in project.find_jobs({'voltage': v, 'case': case, 'seed': seed}):
                copy_file('system_lmp.gro', job.id, v, case, seed)
                copy_file('sample.data', job.id, v, case, seed)
                print('{} done {}'.format(i, job.id))
                i += 1