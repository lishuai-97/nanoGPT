# config for training GPT-2 (345M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_345M.py

exp_name = 'gpt2-345M-lr-1e-3'  # should be same as the run_gpt2.sh script and be changed for each new experiment

wandb_log = False
wandb_project = 'owt'
wandb_run_name=exp_name

tensorboard_log = True
tensorboard_run_name=exp_name

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 32
block_size = 1024
gradient_accumulation_steps = 8


# model configs
n_layer = 24
n_head = 16
n_embd = 1024


# this makes total number of tokens be 300B
learning_rate = 1e-3
max_iters = 600000
lr_decay_iters = 600000
min_lr = 1e-4

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 1

# weight decay
weight_decay = 1e-2
