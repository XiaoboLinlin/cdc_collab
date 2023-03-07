from flow import FlowProject
import signac
import flow
# import environment_for_rahman
import environment_pbs

import MDAnalysis as mda
import mdtraj as md
import numpy as np

def workspace_command(cmd):
    """Simple command to always go to the workspace directory"""
    return " && ".join(
        [
            "cd {job.ws}",
            cmd if not isinstance(cmd, list) else " && ".join(cmd),
            "cd ..",
        ]
    )

sample_file = "sample.gro"
unwrapped_file = 'sample_unwrapped.xtc'
init_file = "system.gro"
for_lammps_data = "sample.data"
conp_file = "restart.final"
lammps_init_file =  "sample.data"
restart_file = 'file.restart.100000'

class Project(FlowProject):
    pass

@Project.label
def run_cpmed(job):
    return job.isfile(conp_file)

@Project.operation
@Project.pre.isfile(lammps_init_file)
@Project.post.isfile(restart_file)
@flow.cmd
def run_cpm(job):
    return _lammps_str(job)

def _lammps_str(job, 
                if_restart=0, 
                if_wat =0, 
                in_path = 'lammps_input/in.data_gcc', 
                exe='/global/cfs/cdirs/m1046/Xiaobo/installed_software/lammps_jan_3/build_gcc/lmp_mpi'):
    
    """Helper function, returns lammps command string for operation 
        Note: need to use cori_start.sh or cori_repeat.sh according to demand
    """
    # exe = '/global/cfs/cdirs/m1046/Xiaobo/installed_software/lammps_May272021_CPM_cray/build_new/lmp_mpi'
    case = job.statepoint()["case"] 
    voltage = job.statepoint()["voltage"] 
    lammps_input = signac.get_project().fn(in_path)
    temperature = 400
    print(case)
    print(job.id)
    if case == 'wat_litfsi':
        if_wat = 1
    
    print('if_restart is ', if_restart)
    
    cmd ='srun --exclusive --ntasks=32 {exe} -in {input} '\
        '-var voltage {voltage} '\
        '-var if_restart {if_restart} '\
        '-var if_wat {if_wat} '\
        '-var temperature {temperature}'
        
    return workspace_command(cmd.format(case = case, voltage=voltage, if_restart=if_restart, if_wat= if_wat, temperature = temperature, exe = exe, input = lammps_input))


@Project.label
def rerun_cpmed(job):
    return job.isfile(conp_file)

@Project.operation
@Project.pre.isfile(restart_file)
@Project.post.isfile(conp_file)
@flow.cmd
def rerun_cpm(job):
    return _lammps_str(job, if_restart=1)


q_file = "q.txt"
@Project.label
def clean_save_q(job):
    return job.isfile(q_file)

result_file = 'restart.final'
# tmp_result_file = 'file.restart.5000000'

@Project.operation
@Project.pre.isfile(result_file)
@Project.post.isfile(q_file)
def run_clean_save_q(job):
    ### clean original dump file and save q
    
    ### step 1: clean original dump file: delete overlaped frames and the last frame
    import time
    from CPManalysis.clean_file import clean_dumpfile
    from CPManalysis.read_file import q_np
    tic = time.perf_counter()
    print(job.id,flush=True)
    print("start to clean and create new new_conp.lammpstrj, 2.5 min", flush=True)
    original_trj_file = os.path.join(job.ws, 'conp.lammpstrj')
    
    new_content = clean_dumpfile(original_trj_file)

    trj_file = os.path.join(job.ws, 'new_conp.lammpstrj')
    text_file = open(trj_file, "w")
    n = text_file.write(new_content)
    text_file.close()
    toc = time.perf_counter()
    print(f"total time of clean conp.lammpstrj is {toc - tic:0.4f} seconds", flush=True)
    
    ### step 2: save q to .npy file
    toc = time.perf_counter()
    print('start to save charge.npy file, 7 min', flush=True)
    
    top_file = os.path.join(job.ws, 'system_lmp.gro')
    u = mda.Universe(top_file)
    n_atom = u.atoms.n_atoms
    charge_2d = q_np(trj_file, n_atom)
    charge_file = os.path.join(job.ws, 'charge.npy')
    np.save(charge_file, charge_2d)
    np.savetxt(os.path.join(job.workspace(), 'q.txt'), [1,1])
    print('it is done', flush=True)
    toc = time.perf_counter()
    print(f"total time of saving q is {toc - tic:0.4f} seconds", flush=True)

conpxtc_file = 'xtc.txt'
@Project.label
def save_xtc(job):
    return job.isfile(conpxtc_file)

new_conp_file = 'q.txt'
@Project.operation
@Project.pre.isfile(new_conp_file)
@Project.post.isfile(conpxtc_file)
def run_save_xtc(job):
    ### save .xtc file using new_conp.lammpstrj file
    print('start to save conp.xtc file, 25 min', flush=True)
    trj_file = os.path.join(job.ws, 'new_conp.lammpstrj')
    top_file = os.path.join(job.ws, 'system_lmp.gro')
    trj = md.load(trj_file, top=top_file)
    save_file = os.path.join(job.ws, 'new_conp.xtc')
    trj.save(save_file)
    np.savetxt(os.path.join(job.workspace(), 'xtc.txt'), [1,1])
    print('it is done', flush=True)
    

total_save_file = "total_save_done.txt"
@Project.label
def total_save(job):
    return job.isfile(total_save_file)

result_file = 'restart.final'
@Project.operation
@Project.pre.isfile(result_file)
@Project.post.isfile(total_save_file)
def run_total_save(job):
    # print(job.id)
    ### step 1: produce cleaned lammpstrj with original charge(new_conp.lammpstrj) and save original charge info to charge.npy file.
    run_clean_save_q(job)
    # ### step 2: save xtc file
    run_save_xtc(job)
    ### resulting production file: charge.npy, charge_modified.npy, new_conp.lammpstrj, new_conp_modified.lammpstrj
    np.savetxt(os.path.join(job.workspace(), 'total_save_done.txt'), [1,1])
    

@Project.label
def save_pele_q(job):
    return job.isfile(pele_q_file)

pele_q_file = "pele_q_file.txt"
@Project.operation
@Project.pre.isfile(q_file)
@Project.post.isfile(pele_q_file)
def run_save_pele_q(job):
    ### produce new charge.npy files with discarded frames and atom charge summation calculation for positive electrode
    print('start to save pele_q', flush=True)
    charge_file = os.path.join(job.workspace(), "charge.npy")
    charge = np.load(charge_file)
    # if job.statepoint()['seed'] ==3:
    #     discard_frame = 2000
    # else:
    #     discard_frame = 50
    # charge = charge[discard_frame:]
    
    ### new_charge.npy is the atom charge after discarded frames
    # new_charge_file = os.path.join(job.ws, 'new_charge.npy')
    # np.save(new_charge_file, charge)
    
    gro_file = os.path.join(job.workspace(), "system_lmp.gro")
    gro_trj = md.load(gro_file)
    
    pos_ele = gro_trj.top.select('residue 1') ## IMportant: residue in mdtraj equal to resid in VMD and in gro file
    pos_trj = gro_trj.atom_slice(pos_ele)
    pos_charge = charge[:,pos_ele]
    sum_pos_q = np.sum(pos_charge, axis =1)
    xdata = np.arange(0, len(sum_pos_q),1)
    xdata = xdata * 0.002
    post_pele_charge = np.stack((xdata, sum_pos_q), axis = 1)
    
    ### pele_charge.npy is summed charge in positive electrode
    pele_charge_file = os.path.join(job.ws, 'pele_charge.npy') ### 
    np.save(pele_charge_file, post_pele_charge)
    np.savetxt(os.path.join(job.workspace(), 'pele_q_file.txt'), [1,1])
    print('it is done')

    


if __name__ == "__main__":
    Project().main()
