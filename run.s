#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=torch_farris
#SBATCH --mail-type=END
#SBATCH --mail-user=fda239@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load cuda/11.3.1 
source activate /scratch/fda239/penv

cd /scratch/fda239/Capstone2021
python train.py