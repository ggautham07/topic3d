#!/bin/bash
#SBATCH --account=c1916693
#SBATCH --job-name="topic3d_test_Hung-HL_array"
#SBATCH --constraint=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-32
#SBATCH --exclusive
#SBATCH --time=00:15:00

module purge
module load python anaconda3
conda activate chromtopic-test-nodask

export MPLBACKEND=agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cmd="topic3d-run"
data="/scratch/CT3/c1916693/gganesh/topics/data"

echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "Running on host: $(hostname)"
echo "Data folder: ${data}"

# # Build list of files matching
# files=( "${data}"/hung*Forelimb* )
# n_files=${#files[@]}
# echo "Found ${n_files} files matching ${data}/hung*"

# # Safety check: ignore tasks beyond number of files
# if (( SLURM_ARRAY_TASK_ID >= n_files )); then
#     echo "No file for task id ${SLURM_ARRAY_TASK_ID} (n_files=${n_files}), exiting."
#     exit 0
# fi

# echo "Task ${SLURM_ARRAY_TASK_ID} will process file: ${file}"

srun topic3d-run \
    --data_file /mnt/PALM_dataserv/DATA/Gautham/Topics/processed/data/hung2024BoundaryStackingInteractions_Mm-Forelimb-Pitx1.h5 \
    --results_base_dir /mnt/PALM_dataserv/DATA/Gautham/Topics/training/ \
    --save_logs \
    --tag "hung2018_tests" \
    --proximity_threshold 150 \
    --min_efficiency 60 \
    --pwd_normalization "R(s)" \
    --imputed \
    --num_components 10 \
    --max_iter 10 \
    --train_data_proportion 1
