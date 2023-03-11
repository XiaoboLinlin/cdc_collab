{% extends "base_script.sh" %}
{% block header %}
#!/bin/bash
#SBATCH --job-name={{id}}
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --time={{ walltime|format_timedelta }}
#SBATCH --error=./error/error_%j.err
#SBATCH --output=./output/run_out_%j.log 

conda activate myenv

{% endblock %}