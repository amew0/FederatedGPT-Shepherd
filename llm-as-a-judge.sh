#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=judge
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --account=kunf0097
# SBATCH --mem=80000MB
#SBATCH --output=./out/%j.out
#SBATCH --error=./out/%j.err
# SBATCH --nodelist=gpu-10-4

 
module load miniconda/3
conda activate torch20
echo "Finally - out of queue" 
python llm-as-a-judge.py