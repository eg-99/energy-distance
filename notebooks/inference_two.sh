#!/bin/sh
#
#SBATCH --account=edu           # The account name for the job.
#SBATCH --job-name=mteb-eval_hotpotqa          # The job name.
#SBATCH -c 2                    # The number of cpu cores to use.
#SBATCH --time=5:00:00            # The time the job will take to run
#SBATCH --mem-per-cpu=8gb	# The memory the job will use per cpu core.

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MODIFY THE FOLLOWING FOR DIFFERENT CONFIGURATIONS
#SBATCH --nodes=1               # Number of nodes
#SBATCH --gres=gpu:2           # Number of GPUs per node
#SBATCH --ntasks-per-node=1     # Number of processes per node

# module load anaconda
module purge
module load cuda/12.4 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
source ~/.bashrc
source /insomnia001/depts/edu/users/eg3346/anaconda3/etc/profile.d/conda.sh
conda activate testenv           # Activate the virtual environment
nvidia-smi

# Set PYTHONPATH to prioritize the virtual environment
export PYTHONPATH=$CONDA_PREFIX/lib/python3.9/site-packages:$PYTHONPATH
pip list
#export CUDA_VISIBLE_DEVICES=1

srun python /insomnia001/depts/edu/users/eg3346/energy-distance/notebooks/train_sbert_coir_2.py
#srun python /moto/home/ggn2104/cos_sim/eval_dataset_subset_length.py 

# End of script
