#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=ft-medical
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --account=kunf0097
# SBATCH --mem=80000MB
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --nodelist=gpu-11-3
 
module load miniconda/3
conda activate torch20
echo "Finally - out of queue" 
CUDA_VISIBLE_DEVICES=0,1 python eval.py
# python eval.py