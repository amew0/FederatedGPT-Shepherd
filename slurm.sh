#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=shepherd
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --account=kunf0007
#SBATCH --mem 490000MB
#SBATCH --output=circle_area.%j.out
#SBATCH --error=circle_area.%j.err
 
module load miniconda/3
conda activate llama2
echo "started" 
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py --global_model 'chavinlo/alpaca-native' --data_path  "./data"  --output_dir  '/dpc/kunf0007/amine/lora-shepherd-7b/' --num_communication_rounds 1  --num_clients  1 --cutoff_len 64 --lora_r 4