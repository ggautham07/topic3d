#!/bin/bash
#SBATCH --account=c1916693
#SBATCH --job-name="test_topic3d_takei_chr17"
#SBATCH --constraint=GENOA
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:20:00

#source /opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh
module purge

# activate environments
#module load cpe/24.07
#module load craype-x86-genoa
#module load PrgEnv-cray
module load python  conda 
conda activate chromtopic-test

#export MPICH_GPU_SUPPORT_ENABLED=1
#export OMP_NUM_THREADS=192  # or whatever --cpus-per-task you choose
#export MKL_NUM_THREADS=192

cmd="/lus/home/CT3/c1916693/mnollmann/Repositories/topic3d/cines/code/launch.py"
data="/scratch/CT3/c1916693/mnollmann/topics"

echo 'Will run '$cmd' at '$data

srun --ntasks-per-node=1 --cpus-per-task=192 --threads-per-core=1 \
    $cmd \
    --root_dir $data \
    --tag takei2021SinglecellNuclearArchitecture_Mm-brain-chr17_wt \
    --num_components 15 \
    --num_jobs 1 \
    --threshold 150 \
    --hardness 1 \
    --learning_method batch \
    --max_iter 500
