#!/bin/bash

#SBATCH --job ml_project
#SBATCH --nodes=1
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1

echo Running on `hostname`
echo workdir $PBS_O_WORKDIR

cd $SLURM_SUBMIT_DIR

#scratch drive folder to work in
SCRDIR=/scr/${SLURM_JOB_ID}

#if the scratch drive doesn't exist (it shouldn't) make it.
if [[ ! -e $SCRDIR ]]; then
        mkdir $SCRDIR
fi

chmod +rX $SCRDIR

echo scratch drive ${SCRDIR}

rsync -rv -q $SLURM_SUBMIT_DIR/ ${SCRDIR}

if [[ -e $SCRDIR/saved_models ]]; then
        rm -rf $SCRDIR/saved_models
fi

cd ${SCRDIR}
module load cuda
conda run --live-stream -n pt1102 python -u lightning.py configs/hrtransformer.yaml

rsync -rv -q $SCRDIR/ $SLURM_SUBMIT_DIR
rm -rf ${SCRDIR}
