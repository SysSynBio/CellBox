

import os, shutil, sys

name = "cyano_rna_slurm"
dispatches = list(range(20))

account = "hyperbio"
time = "4-0" 
node = '1'
core = "1" 
out = "cellbox_out"
err = "cellbox_err"
mail = "song.feng@pnnl.gov"
modules = ["gcc/11.2.0", "python/miniconda22.11"]
extras = [
        "source /share/apps/python/miniconda-py310_22.11.1-1/etc/profile.d/conda.sh", 
        "conda activate cellbox",
        "export PATH=/people/feng626/.conda/envs/cellbox/bin:$PATH",
        "cd /people/feng626/phenome/SystemModeling/CellBox",
        ]


slurm = "#!/bin/sh\n#SBATCH -A " + account + "\n#SBATCH -t " + time + "\n#SBATCH -N " + str(node) + "\n#SBATCH -n " + str(core)

for dispatch in dispatches:
    slurm += "\n#SBATCH -o " + out + '_' + str(dispatch) + ".txt\n#SBATCH -e " + err + '_' + str(dispatch) + ".txt\n#SBATCH --mail-user=" + mail + "\n#SBATCH --mail-type " + "END" + "\n \nmodule purge\n"
    for module in modules:
        slurm += "module load " + module + '\n \n'
    for extra in extras:
        slurm += extra + '\n \n'

    slurm += "max = 50\nfor i in `seq 0 $max`\ndo\n    python scripts/main.py -config=cyano_rna/cyano_rna_config.json -i=$i\ndone\n"

    filename = name + "_rep" + str(dispatch) + ".sbatch"
    with open(filename, 'w') as sbatch:
        sbatch.write(slurm)


