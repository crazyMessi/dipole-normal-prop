#!/bin/bash

# 如果有参数 --model_name 则使用参数指定的模型名
# 否则使用默认的模型名
export MODEL_NAME=$1

export CUDA_VISIBLE_DEVICES=2

export BASE_PATH=$(cd ../; pwd)
echo $BASE_PATH
export PYTHONPATH=$BASE_PATH

export MODEL_BASE_PATH=$BASE_PATH/data

python -u $BASE_PATH/orient_simple.py \
--pc $MODEL_BASE_PATH/$MODEL_NAME \
--export_dir $BASE_PATH/demos/$MODEL_NAME \
--estimate_normals