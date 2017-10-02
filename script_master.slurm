#!/bin/bash -l
#SBATCH --job-name DAve_master
#SBATCH --array=0
#SBATCH --time=00:10:00
#SBATCH -o logs/output.%a.%j.out
#SBATCH -e logs/output.%a.%j.err
#SBATCH --mem-per-cpu=4096
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=1

module purge
module load slurm
module load applications-extra
module load anaconda3/4.2.0

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd $DIR

python3 /scratch/dragon/amd/mishchk/master.py