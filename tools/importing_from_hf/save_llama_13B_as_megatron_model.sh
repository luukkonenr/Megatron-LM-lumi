#!/bin/bash

TARGET_TP=4
TARGET_PP=2
HF_FORMAT_DIR=/scratch/project_462000086/viking-v2/converted_models/viking_v2_13B_iter_0289000_bfloat16/
MEGATRON_FORMAT_DIR=/scratch/project_462000086/risto/viking-v3/v2_converted_checkpoints/viking_v2_13B_iter_0289000_bfloat16/
TOKENIZER_MODEL=$HF_FORMAT_DIR
ITERATION=1

python3 tools/checkpoint/util.py \
  --model-type GPT \
  --loader loader_llama2_hf \
  --saver megatron \
  --target-tensor-parallel-size ${TARGET_TP} \
  --target-pipeline-parallel-size ${TARGET_PP} \
  --load-dir ${HF_FORMAT_DIR} \
  --save-dir ${MEGATRON_FORMAT_DIR} \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --iteration ${ITERATION}