from flow import FlowProject
import signac
import flow
import environment_for_rahman
# import environment_pbs
import MDAnalysis as mda
import mdtraj as md
import numpy as np
import os

def workspace_command(cmd):
    """Simple command to always go to the workspace directory"""
    return " && ".join(
        [
            "cd {job.ws}",
            cmd if not isinstance(cmd, list) else " && ".join(cmd),
            "cd ..",
        ]
    )

init_file = "system.gro"
sample_file = "sample.gro"
unwrapped_file = 'sample_unwrapped.xtc'
conp_file = "restart.final"
lammps_init_file =  "sample.data"
restart_file = 'restart.1000'

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
                in_path = 'lammps_input/in.data_gcc_test', 
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

    cmd ='srun --exclusive --ntasks=64 {exe} -in {input} '\
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
def clean_save_q_xtc(job):
    return job.isfile(q_file)

result_file = 'restart.final'
# tmp_result_file = 'file.restart.5000000'

@Project.operation
@Project.pre.isfile(result_file)
@Project.post.isfile(q_file)
def run_clean_save_q_xtc(job):
    ### clean original dump file and save q
    
    ### step 1: clean original dump file: delete overlaped frames and the last frame
    with job:
        import time
        from CPManalysis.clean_file import clean_dumpfile
        from CPManalysis.read_file import q_np2
        tic = time.perf_counter()
        print(job.id,flush=True)
        print("start to clean and create new new_ele.lammpstrj, 2.5 min", flush=True)
        original_trj_file, trj_file = 'ele.lammpstrj', 'new_ele.lammpstrj'
        new_content = clean_dumpfile(original_trj_file, keep_last_frame=True)
        text_file = open(trj_file, "w")
        n = text_file.write(new_content)
        text_file.close()
        toc = time.perf_counter()
        print(f"total time of clean conp.lammpstrj is {toc - tic:0.4f} seconds", flush=True)
        
        ### step 2: save q to .npy file
        toc = time.perf_counter()
        print('start to save charge.npy file, 7 min', flush=True)
        top_file, charge_file = 'system_lmp.gro', 'charge.npy'
        u_gro = mda.Universe(top_file)
        u_ele = u_gro.select_atoms('resid 1')
        n_atom_ele = u_ele.n_atoms * 2 # *2 because two electrodes
        charge_2d = q_np2(trj_file, n_atom_ele)
        np.save(charge_file, charge_2d)
        print('it is done', flush=True)
        toc = time.perf_counter()
        print(f"total time of saving q is {toc - tic:0.4f} seconds", flush=True)
        
        ### step 3: generate combined nonoverlaped xtc files
        u = mda.Universe('system_lmp.gro', 'conp_0.xtc','conp_1.xtc', continuous=True)
        # write combined xtc file
        u_all = u.select_atoms("all")
        with mda.Writer("conp_total.xtc", u_all.n_atoms) as W:
            for ts in u.trajectory:
                W.write(u_all)
            
        np.savetxt(os.path.join(job.workspace(), 'q.txt'), [1,1])
    
conp_unwrapped_file =  'conp_prepare_done.txt'
@Project.label
def prepared(job):
    return job.isfile(conp_unwrapped_file)

@Project.operation
@Project.pre.isfile(q_file)
@Project.post.isfile(conp_unwrapped_file)
def prepare(job):
    from CPManalysis.gromacs import make_comtrj
    xtc_file = os.path.join(job.workspace(), 'conp_total.xtc')
    gro_file = os.path.join(job.workspace(), 'system_lmp.gro')

    if os.path.isfile(xtc_file) and os.path.isfile(gro_file):
        unwrapped_trj = os.path.join(job.workspace(),
        'conp_unwrapped.xtc')
        os.system('echo 0 | gmx trjconv -f {0} -o {1} -s {2} -pbc nojump'.format(xtc_file, unwrapped_trj, gro_file))

        unwrapped_com_trj = os.path.join(job.ws,'conp_com_unwrapped.xtc')

        print('start to load')
        trj = md.load(unwrapped_trj, top=gro_file)
        comtrj = make_comtrj(trj)
        comtrj.save_xtc(unwrapped_com_trj)
        comtrj[-1].save_gro(os.path.join(job.workspace(),
             'com.gro'))
        np.savetxt(os.path.join(job.workspace(), 'conp_prepare_done.txt'), [1,1])
     

if __name__ == "__main__":
    Project().main()
