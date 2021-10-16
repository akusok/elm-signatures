#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=project_2001220
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:v100:1
#SBATCH --output="test-%j.out"
#SBATCH --error="test-%j.err"

module purge
module load gcc/9.1.0
module load cuda/11.1.0 
module load module_cuml

set -xv
srun python $1
