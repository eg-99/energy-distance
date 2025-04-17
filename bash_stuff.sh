#!/bin/bash

USERNAME=$(whoami)

mkdir /insomnia001/depts/edu/users/$USERNAME
cd /insomnia001/depts/edu/users/$USERNAME

wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh

bash Anaconda3-2024.10-1-Linux-x86_64.sh -b -p /insomnia001/depts/edu/users/$USERNAME/anaconda3 
PATH="/insomnia001/depts/edu/users/$USERNAME/anaconda3/bin:/insomnia001/depts/edu/users/$USERNAME/anaconda3/sbin:$PATH"
CONDA_PYTHON_EXE=/insomnia001/depts/edu/users/$USERNAME/anaconda3/bin/python


conda create --name myenv39 python=3.9
conda init
bash
conda activate myenv39
pip install --upgrade pip --index-url https://pypi.org/simple
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
git clone git@github.com:gnatesan/sentence-transformers-3.4.1.git
git clone git@github.com:gnatesan/mteb-1.34.14.git
pip install -e /path_to_sentence-transformers/sentence-transformers-3.4.1
pip install -e /path_to_mteb/mteb-1.34.14
git clone git@github.com:gnatesan/beir.git


conda create --name testenv python=3.9
conda activate testenv
pip install --upgrade pip --index-url https://pypi.org/simple
pip install sentence-transformers==3.4.1
pip install mteb==1.34.14
git clone git@github.com:gnatesan/energy-distance

chmod +x energy-distance/notebooks/inference_CosSim.sh

srun --pty -t 0-2:00 -A edu /bin/bash
srun --pty -t 0-01:00 --gres=gpu:2 -A edu /bin/bash
# module load cuda

# vim energy-distance/notebooks/inference_CosSim.sh

#!/bin/sh
#
#SBATCH --account=edu           # The account name for the job.
#SBATCH --job-name=mteb-eval_hotpotqa          # The job name.
#SBATCH -c 2                    # The number of cpu cores to use.
#SBATCH --time=48:00:00            # The time the job will take to run
#SBATCH --mem-per-cpu=5G        # The memory the job will use per cpu core.

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MODIFY THE FOLLOWING FOR DIFFERENT CONFIGURATIONS
#SBATCH -N 2               # Number of nodes
#SBATCH --gres=gpu:1            # Number of GPUs per node
#SBATCH --ntasks-per-node=1     # Number of processes per node



# sbatch energy-distance/notebooks/inference_CosSim.sh



# python stuff.py --args
