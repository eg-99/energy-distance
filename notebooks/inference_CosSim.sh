#!/bin/sh
#
#SBATCH --account=edu           # The account name for the job.
#SBATCH --job-name=mteb-eval_hotpotqa          # The job name.
#SBATCH -c 2                    # The number of cpu cores to use.
#SBATCH --time=0-48:00:00            # The time the job will take to run
#SBATCH --mem-per-cpu=5G	# The memory the job will use per cpu core.

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MODIFY THE FOLLOWING FOR DIFFERENT CONFIGURATIONS
#SBATCH -N 1               # Number of nodes
#SBATCH --gres=gpu:2

module load anaconda
module load cuda

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
source ~/.bashrc
conda activate testenv           # Activate the virtual environment
nvidia-smi

# Set PYTHONPATH to prioritize the virtual environment
export PYTHONPATH=$CONDA_PREFIX/lib/python3.9/site-packages:$PYTHONPATH

#export CUDA_VISIBLE_DEVICES=1

srun python ./energy-distance/notebooks/eval_dataset.py
#srun python /moto/home/ggn2104/cos_sim/eval_dataset_subset_length.py 

# End of script
