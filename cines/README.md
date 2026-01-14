# Quick start guide for installing and running `topic3d` on Adastra

Parts of this guide have been borrowed from Adastra's [official quick start guide](https://dci.dci-gitlab.cines.fr/webextranet/quick_start/index.html)

## Connection to the server

After the username and password are made available, login with

```
<username>@adastra.cines.fr
```

### Setup SSH key

Edit `~/.ssh/config` on the computer being used to connect to Adastra

```
Host adastra
  HostName adastra.cines.fr
  User <username>
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 10m
```

And then connect using

```
$ ssh adastra
```

### Viewing project details

Many details pertaining to the project account on Adastra can be viewed using the `myproject` module. Some examples are given below, and more information is available in the [official documentation](https://dci.dci-gitlab.cines.fr/webextranet/user_support/index.html#login-unique)

`myproject --list (-l)`   --->  to list all associated projects
`myproject --state (-s)`  --->  current status of the project, space and file usage, etc.
`myproject --cines (-c)`  --->  list all CINES project directories

### Copying data to Adastra

```bash
rsync --progress -avh --ignore-existing <path_to_local_files> <username>@adastra.cines.fr:<path_to_copy_to_server>
```

More information is available [here](https://dci.dci-gitlab.cines.fr/webextranet/data_storage_and_transfers/index.html#data-transfer)


## Setting up the environment

### Load required modules

Adastra uses modules that can be loaded when needed. The complete list of modules filtered by a search string (leave empty for no filtering) can be viewed by running

```
module spider <string>
```

For topic decomposition, Python and Anaconda must be loaded.

```
module load python anaconda3
```

And the list of currently loaded modules can be listed using

```
module list
```

More information about modules is available [here](https://dci.dci-gitlab.cines.fr/webextranet/user_support/index.html#module-why-and-how)


### Installing the virtual environment

This repository must first be cloned since it contains the required scripts and environment files.

```bash 
mkdir -m 755 $HOMEDIR/repositories
git clone https://github.com/ggautham07/topic3d.git --branch cines
cd topic3d
```

The environment can then be installed in <path_to_env> (preferably in `$WORKDIR`) from the file `cines/chromtopic-nodask.yml`

```bash
conda env create --prefix <path_to_env> --file cines/chromtopic-nodask.yml
```

> [!note] Possible incompatibility with Adastra's anaconda3 module
> In some cases, installing the environment with a file fails due to reasons not well understood. In this case, a suitable environment can still be created  with the base Python packages and pip.

```bash
conda env create --prefix <path_to_env> python=3.13.5
```

To include the parent directory of <path_to_env> (for instance, if it is used to store conda environments) in the list of directories that conda surfs to prepare the list of environments, try

```bash
conda config --append envs_dirs <path_to_envs_folder>
```

This will ensure that the environment is usable and can be detected by conda.

```bash
conda info --envs
``` 

Load the environment.

```bash
conda activate chromtopic-nodask        # replace chromtopic-nodask with <path_to_env> if the environment was not listed in the previous step
```

Install `topic3d` in this environment using pip. In case a minimal environment was created earlier, `topic3d`'s dependencies should still be correctly installed. Ensure that the correct `pip` on the system is being used by running `which -a pip`.

```bash
cd $HOMEDIR/repositories/topics3d/
pip install .
```

Verify the installation using `conda list`.


## Running parallel topic decomposition(s)

### Dataset storage

It is highly recommended to store datasets in `$SCRATCHDIR`. However, note that they are automatically cleared every thirty days.

Datasets used now are being stored in `$SHAREDSCRATCHDIR/topics/data`, with some testing copies in `/lus/scratch/CT3/c1916693/gganesh` (i.e., Gautham's `$SCRATCHDIR`)

### Batch script model

This model in available in `topic3d/cines/example_run.sh`. It is used to run parallel topic decompositions for a specific dataset, for 8 different `num_components`, with 24 tasks each in parallel (hence totalling 192 trainings). The values given here are necessary for the full exclusive usage of a node with 192 computing units (hence, one CPU per training)

```
#!/bin/bash
#SBATCH --account=c1916693
#SBATCH --job-name=<job_name>
#SBATCH --constraint=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --ntasks-per-node=192
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/lus/home/CT3/c1916693/gganesh/slurm-outfiles/%j.out
#SBATCH --error=/lus/home/CT3/c1916693/gganesh/slurm-outfiles/%j.out

set -eu    # for error handling

module purge
module load python anaconda3
conda activate chromtopic-nodask

export MPLBACKEND=agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cmd=$HOMEDIR"/.local/bin/topic3d-run"
data=$SCRATCHDIR"/topics/data/<filename.h5>"
outdir=$SCRATCHDIR"/topics/training"

echo "Host: "$(hostname)
echo "Data file path: "${data}
echo "Base output directory: "${outdir}

num_components=(5 10 15 20 25 30 35 40)

for nc in "${num_components[@]}";
do
    echo "Launching trainings with num_components="${nc}
    srun --ntasks=24  ${cmd} \
         --data_file ${data} \
         --results_base_dir ${outdir} \
         --save_logs \
         --proximity_threshold 150 \
         --min_efficiency 60 \
         --num_components ${nc} \
         --max_iter 1000 \
        --train_data_proportion 1 &
done

wait

echo "All operations completed."
```

Scripts using job arrays for massive parallelization are being written and will be updated in future iterations.
