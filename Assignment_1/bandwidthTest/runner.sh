#!/bin/bash
#SBATCH -J mawass_lab_1
#SBATCH -t 00:05:00
#SBATCH -A edu19.DD2360
#SBATCH --nodes=1
#SBATCH -C Haswell 
#SBATCH --gres=gpu:K420:1
#SBATCH -e error_file.e

srun -n 1 ./bandwidthTest --mode=shmoo > outputfile.txt
