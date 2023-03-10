#!/bin/bash
#SBATCH -A hyperbio
#SBATCH -t 4-0
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -o cellbox_out
#SBATCH -e cellbox_err
#SBATCH --mail-user=song.feng@pnnl.gov
#SBATCH --mail-type END

module purge
module load gcc/11.2.0
module load python/miniconda22.11

conda activate cellbox

export PATH=/people/feng626/.conda/envs/cellbox/bin:$PATH

cd /people/feng626/phenome/SystemModeling/CellBox

python scripts/main.py -config=cyano_rna/cyano_rna_config.json -i=230309
