#!/bin/bash
#SBATCH -J mawass
#SBATCH -t 00:05:00
#SBATCH -A edu19.DD2360
#SBATCH --nodes=1
#SBATCH -C Haswell 
#SBATCH --gres=gpu:K420:1
#SBATCH -e error_file.e

srun -n 1 ./exercise_1.out > outfile.txt
