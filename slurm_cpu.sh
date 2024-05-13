#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=shepherd
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --account=kunf0007
#SBATCH --output=%j_cpu.out
#SBATCH --error=%j_cpu.err
 
module load miniconda/3
conda activate llama2
echo "Finally - out of queue" 
python ./ft.py