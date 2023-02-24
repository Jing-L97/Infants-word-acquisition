#!/bin/bash
#SBATCH --job-name=char_no_lstm
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --cpus-per-task=10 
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores 
#SBTACH --gres=gpu:1   
#SBATCH --time=30:00:00
#SBATCH --output=lstm-char_no_bound-3%j.out # output file name
#SBATCH --error=lstm-char_no_bound-3%j.err  # error file name

echo "Running job on $(hostname)"
source /shared/apps/anaconda3/etc/profile.d/conda.sh


module purge
conda activate temporal


fairseq-train --task language_modeling \
      fairseq_bin_data \
      --task language_modeling \
      --save-dir checkpoints-trial \
      --keep-last-epochs 2 \
      --tensorboard-logdir tensorboard \
      --arch lstm_lm \
      --decoder-embed-dim 200 --decoder-hidden-size 1024 --decoder-layers 3 \
      --decoder-out-embed-dim 200 \
      --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
      --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 1000 --warmup-init-lr 1e-07 \
      --dropout 0.1 --weight-decay 0.01 \
      --sample-break-mode none --tokens-per-sample 2048 \
      --max-tokens 163840 --update-freq 1 --max-update 100000
