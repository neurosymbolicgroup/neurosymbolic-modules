#!/bin/sh
#
#SBATCH --job-name=arc_simon
#SBATCH --output=arc_simon_12_20.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1000

srun bash run_arc.sh
