#!/bin/bash
#SBATCH -J STAGE_0
#SBATCH -t 00:20:00
#SBATCH -A edu19.DD2360
#SBATCH --nodes=1
#SBATCH -C Haswell 
#SBATCH --gres=gpu:K420:1
#SBATCH -e error_file.e
#SBATCH --mail-type=END
#SBATCH --mail-user=mawass
#SBATCH --output=output.txt

mpirun -n 1 bin/miniWeather
