# Install and run topic decompositions in cines


## connection


### setup SSH key

edit `~/.ssh/config`

```
Host adastra
  HostName adastra.cines.fr
  User mnollmann
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 10m
```

```
$ ssh adastra
```

### copying data to CINES

rsync --progress -avh /mnt/grey/DATA/users/gautham/Workspace/topics/data/processed.tar.gz mnollmann@adastra.cines.fr:/work/CT3/c1916693/SHARED

### modules

#### load modules

To start with we need to load the right environments:

```
ml python conda
```


### conda installation



```bash 
cd Repositories
git clone git@github.com:ggautham07/topic3d.git
cd topic3d
```

Then run:
```bash
conda env create -f chromtopic-test.yml
cd Repository/topics3d/
conda activate chromtopic-test
pip install .
```

### dataset

copied datasets to:

`/work/CT3/c1916693/SHARED/data`

### test run

This is the command to run in a normal computer
```
/work/CT3/c1916693/SHARED/temp/launch.py --root_dir /work/CT3/c1916693/SHARED --tag takei2021SinglecellNuclearArchitecture_Mm-brain-chr17_wt --num_components 15 --num_jobs 1 --threshold 150 --hardness 1 --learning_method batch --max_iter 500
```

### sbatch


```bash
#!/bin/bash
#SBATCH --account=c1916693
#SBATCH --job-name="test_topic3d_takei_chr17"
#SBATCH --constraint=GENOA
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:20:00

module purge

# activate environments
module load python conda
conda activate chromtopic-test

#export MPICH_GPU_SUPPORT_ENABLED=1
#export OMP_NUM_THREADS=192  # or whatever --cpus-per-task you choose
#export MKL_NUM_THREADS=192

srun --ntasks-per-node=1 --cpus-per-task=192 --threads-per-core=1  /work/CT3/c1916693/SHARED/temp/launch.py --root_dir /work/CT3/c1916693/SHARED --tag$



```

