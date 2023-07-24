import os, shutil, sys

name = "central_carbon_ps_de"
dispatch_num = 20
loop_num = 50

account = "ldrd_cheung2022"
time = "4-0" 
node = '1'
core = "1" 
out = "central_carbon_ps_de_out"
err = "central_carbon_ps_de_err"
# mail = "song.feng@pnnl.gov"
modules = ["gcc/11.2.0", "python/miniconda23.3.1"]
extras = [
        "source /share/apps/python/miniconda23.3.1/etc/profile.d/conda.sh", 
        "conda activate cellbox",
        "export PATH=/people/feng626/.conda/envs/cellbox/bin:$PATH",
        ]
workdir = "cd /people/feng626/phenome/SystemModeling/CellBox"

slurm_template = "#!/bin/sh\n#SBATCH -A " + account + "\n#SBATCH -t " + time + "\n#SBATCH -N " + str(node) + "\n#SBATCH -n " + str(core)

for dispatch in range(dispatch_num):
    slurm = slurm_template
    slurm += "\n#SBATCH -o " + out + '_' + str(dispatch) + ".txt\n#SBATCH -e " + err + '_' + str(dispatch) + ".txt\n"
    # slurm += "#SBATCH --mail-user=" + mail + "\n#SBATCH --mail-type END\n"
    slurm += "\nmodule purge\n"
    for module in modules:
        slurm += "module load " + module + '\n'
    slurm += '\n'
    for extra in extras:
        slurm += extra + '\n'
    slurm += '\n'
    slurm += workdir + "\n \n"

    slurm += "max=" + str((dispatch + 1) * loop_num - 1) + "\nfor i in `seq " + str(dispatch * loop_num) + " $max`\ndo\n    python scripts/main.py -config=cyano_small_system/central_carbon_ps_de/central_carbon_ps_de.json -i=$i\ndone\n"

    filename = name + "_rep" + str(dispatch) + ".sbatch"
    with open(filename, 'w') as sbatch:
        sbatch.write(slurm)
    submission = "sbatch " + filename 
    os.system(submission)


