#!/bin/bash

ROOT_DIR=./
EXP_NAME=gpt2-345M-lr-1e-3
TENSORBOARD_PATH=$ROOT_DIR/out/logs/$EXP_NAME

mkdir -p $TENSORBOARD_PATH

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2_345M.py | tee -a ${TENSORBOARD_PATH}/output.log