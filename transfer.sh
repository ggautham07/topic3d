#!/usr/bin/env bash

filename="model.jl"
directories=( /mnt/grey/DATA/users/JB/Topic_analysis/2025_FOF_format/training/Bintu_HCT116_chr21_WT_40_83/2025_05_12/[0-4][0]_components/* )
# echo $directories
for directory in "${directories[@]}"
do
    echo "$directory/$filename"
    dest=./models
    echo "$(basename $directory).jl"
    rsync -av --ignore-existing "$directory/$filename" "$dest/$(basename $directory).jl"
done