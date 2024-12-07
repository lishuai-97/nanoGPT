#!/bin/bash

ROOT_DIR=./
EXP_NAME=gpt2-124M-lr-1e-3
TENSORBOARD_PATH=$ROOT_DIR/out/logs/$EXP_NAME

mkdir -p $TENSORBOARD_PATH

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py config/train_gpt2.py | tee -a ${TENSORBOARD_PATH}/output.log