#!/bin/sh
#SBATCH --partition=gpu-l40s
#SBATCH --job-name=gpu_job           
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=10        
#SBATCH --mem=50G                  
#SBATCH --time=24:00:00            
#SBATCH --output=embedding.log 

nvidia-smi
module purge
module load cuda12.2/toolkit/12.2.2
module load cudnn9.1-cuda12.2/9.1.1.17
module load nccl2-cuda12.2-gcc11/2.22.3


export CUDA_HOME=/cm/shared/apps/cuda12.2/toolkit/12.2.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cm/shared/apps/nccl2-cuda12.2-gcc11/2.22.3/lib:$LD_LIBRARY_PATH


ldd $(python -c "import torch; print(torch.__file__)") | grep nccl


python index_commits.py --model_name grit_instruct --dataset_name AD
python index_query.py --model_name grit_instruct --dataset_name AD

python index_file.py --model_name grit_instruct_512_file --dataset_name AD
python index_query.py --model_name grit_instruct_512_file --dataset_name AD
