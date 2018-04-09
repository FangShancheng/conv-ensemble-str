#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export workdir=workdir-master

python train_eval.py --output_dir=${workdir}      \
                     --train_steps=300000         \
                     --dataset_dir=/home/data/Dataset/tf-mjsynth
