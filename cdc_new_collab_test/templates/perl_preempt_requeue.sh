{% extends "base_script.sh" %}
{% block header %}
#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name={{id}}
#SBATCH -q preempt
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --time={{ walltime|format_timedelta }}
#SBATCH --error=./error/error_%j.err
#SBATCH --output=./output/run_out_%j.log 
#SBATCH --comment=144:00:00  #desired time limit
##SBATCH --signal=B:USR1@60  #sig_time (60 seconds) should match your checkpoint overhead time
##SBATCH --open-mode=append

# specify the command to use to checkpoint your job if any (leave blank if none)
ckpt_command=

# user setting and executables go here

trap 'scontrol requeue ${SLURM_JOB_ID}; exit 15' 15 
# module load python
conda activate myenv

{% endblock %}