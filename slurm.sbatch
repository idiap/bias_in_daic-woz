#!/bin/bash
#SBATCH --job-name=diac-woz
#SBATCH --output=output/diac-woz_%A_%a.out
#SBATCH --error=output/diac-woz_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --nodes 1
#SBATCH --time=3:00:00
#SBATCH --gpus rtx3090:1
#SBATCH --account YOUR_ACCOUNT
#SBATCH --array=1-12

python main.py
