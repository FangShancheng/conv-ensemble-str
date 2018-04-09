#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=''
# MJSynth, SVT, ...
dataset=MJSynth
# continuous_eval, evaluate
schedule=continuous_eval
workdir=workdir-master
split_name=test
eval_step=500

python train_eval.py --output_dir=${workdir}    \
                     --schedule=${schedule}     \
                     --dataset_name=${dataset}  \
                     --eval_steps=${eval_step}  \
                     --split_name=${split_name} \
                     --beam_width=1             \
                     --dataset_dir=/home/data/Dataset/tf-mjsynth