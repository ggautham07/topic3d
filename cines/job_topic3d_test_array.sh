#!/bin/bash
#SBATCH --account=c1916693
#SBATCH --job-name="topic3d_takei_array"
#SBATCH --constraint=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --time=00:25:00
#SBATCH --array=0-9   # <-- adjust this when submitting (see below)

module purge
module load python conda
conda activate chromtopic-test

export MPLBACKEND=agg

cmd="/lus/home/CT3/c1916693/mnollmann/Repositories/topic3d/cines/code/launch.py"
data="/scratch/CT3/c1916693/mnollmann/topics"

echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "Running on host: $(hostname)"
echo "Data folder: ${data}"

# Build list of files matching takei2021*
files=( "${data}"/takei2021* )

n_files=${#files[@]}
echo "Found ${n_files} files matching ${data}/takei2021*"

# Safety check: ignore tasks beyond number of files
if (( SLURM_ARRAY_TASK_ID >= n_files )); then
    echo "No file for task id ${SLURM_ARRAY_TASK_ID} (n_files=${n_files}), exiting."
    exit 0
fi

file="${files[$SLURM_ARRAY_TASK_ID]}"
basename="$(basename "$file")"
tag="${basename%.*}"   # strip extension if present

echo "Task ${SLURM_ARRAY_TASK_ID} will process file: ${file}"
echo "Using tag: ${tag}"

srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --threads-per-core=1 \
    "${cmd}" \
    --root_dir "${data}" \
    --tag "${tag}" \
    --num_components 5,10,15,20,25,30 \
    --num_jobs 32 \
    --threshold 150 \
    --hardness 1 \
    --learning_method batch \
    --max_iter 500
