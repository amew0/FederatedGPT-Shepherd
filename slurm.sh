#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=shepherd
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --account=kunf0007
#SBATCH --mem=80000MB
#SBATCH --output=%j.out
#SBATCH --error=%j.err
 
module load miniconda/3
conda activate llama2
echo "Finally - out of queue" 
python ./ft.py