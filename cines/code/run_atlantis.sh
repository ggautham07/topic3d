#!/bin/bash
#SBATCH --account=c1916693
#SBATCH --job-name="test_topic3d_takei_chr17"
#SBATCH --constraint=GENOA
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:20:00

#module purge

# activate environments
#module load python conda
#mamba activate /home/marcnol/miniforge3/envs/chromtopic-test

#export MPICH_GPU_SUPPORT_ENABLED=1
#export OMP_NUM_THREADS=192  # or whatever --cpus-per-task you choose
#export MKL_NUM_THREADS=192

#cdm="/lus/home/CT3/c1916693/mnollmann/Repositories/topic3d/cines/code/launch.py"
cmd="/home/marcnol/Repositories/topic3d/cines/code/launch.py"
data="/home/marcnol/grey/users/marcnol/test_topics"

echo 'Will run '$cmd' at '$data

$cmd \
    --root_dir $data \
    --tag takei2021SinglecellNuclearArchitecture_Mm-brain-chr17_wt \
    --num_components 15 \
    --num_jobs 1 \
    --threshold 150 \
    --hardness 1 \
    --learning_method batch \
    --max_iter 500

#--num_components 10,15,20,25 --num_jobs 5 --threshold 150 --hardness 1 --learning_method batch --max_iter 500