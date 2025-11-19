 
 # single srun command
 
 srun --time=00:7:00 --exclusive --nodes=1 --constraint=GENOA --job-name="test_topic3d_marion" --account=c1916693 --ntasks-per-node=1 --cpus-per-task=192 --threads-per-core=1 $cmd --root_dir $data  --tag $tag --num_components 30 --num_jobs 192 --threshold 150 --hardness 1 --learning_method batch --max_iter 500