#!/bin/bash
#SBATCH --account=c1916693
#SBATCH --job-name="topic3d_Hung-FL_multi-num-comps"
#SBATCH --constraint=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks=160
#SBATCH --ntasks-per-node=160
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/lus/home/CT3/c1916693/gganesh/slurm-outfiles/%j.out
#SBATCH --error=/lus/home/CT3/c1916693/gganesh/slurm-outfiles/%j.out

set -eu

module purge
module load python anaconda3
conda activate chromtopic-nodask

export MPLBACKEND=agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cmd="topic3d-run"
data=$SCRATCHDIR"/topics/data/hung2024BoundaryStackingInteractions_Mm-Forelimb-Pitx1.h5"
outdir=$SCRATCHDIR"/topics/training"

echo "Host: "$(hostname)
echo "Data file path: "${data}
echo "Base output directory: "${outdir}

num_components=(5 10 15 20 25 30 35 40 45 50)

for nc in "${num_components[@]}";
do
    echo "Launching trainings with num_components="${nc}
    srun --ntasks=16  ${cmd} \
        --data_file ${data} \
        --results_base_dir ${outdir} \
        --save_logs \
        --proximity_threshold 150 \
        --min_efficiency 60 \
        --pwd_normalization "R(s)" \
        --imputed \
        --num_components ${nc} \
        --max_iter 1500 \
        --train_data_proportion 1 &
done

wait

echo "All operations completed."
