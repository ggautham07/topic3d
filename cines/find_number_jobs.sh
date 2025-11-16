#!/bin/bash

echo 'This is used to find how to optimize the number of arrays given to slurm'

data="/scratch/CT3/c1916693/mnollmann/topics"
files=( "${data}"/takei2021* )
n_files=${#files[@]}
echo "Found ${n_files} files"

echo "Now you can run the script using: \n sbatch --array=0-$((n_files-1)) job_topic3d_test_array.sh"
