#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=''
# MJSynth, SVT, ...
dataset=MJSynth
# continuous_eval, evaluate
schedule=evaluate
workdir=workdir-master

python train_eval.py --output_dir=${workdir}   \
                     --schedule=${schedule}    \
                     --dataset_name=${dataset} \
                     --beam_width=5            \
                     --dataset_dir=/home/data/Dataset/tf-mjsynth