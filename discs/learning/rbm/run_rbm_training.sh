#!/bin/bash

dataset=mnist
num_categories=2
num_hidden=200
# SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# save_root=${SCRIPT_DIR}/../../../storage/models/rbm
save_root=$HOME/Workspace/dev/discs/storage/models/rbm

export CUDA_VISIBLE_DEVICES=0
# export XLA_FLAGS="--xla_force_host_platform_device_count=4"

python -m discs.learning.rbm.main_rbm_training \
  --config="exp_config.py:${dataset}-${num_categories}-${num_hidden}" \
  --save_root=$save_root \
  $@