#!/bin/bash
#
# CompecTA (c) 2017
#
# You should only work under the /scratch/users/<username> directory.
#

# -= Resources =-
#
#SBATCH --job-name=maac_auto_gov                   # DON'T FORGET TO UPDATE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

################################################################################
##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################
################################################################################

echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

## Load Modules

module load gcc/8.2.0
module load cuda/11.3
module load python/3.10.4
module load openmpi/4.1.4

echo "==============================================================================="
source activate venv

python main.py test_model --config ./config/simplified_config.json --use_gpu

source deactivate venv
