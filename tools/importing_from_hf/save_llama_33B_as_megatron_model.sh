#!/bin/bash

TARGET_TP=2
TARGET_PP=1
HF_FORMAT_DIR=/scratch/project_462000086/viking-v2/converted_models/viking_v2_7B_iter_0378000_bfloat16/
MEGATRON_FORMAT_DIR=/scratch/project_462000086/risto/viking-v3/v2_converted_checkpoints/viking_v2_7B_iter_0378000_bfloat16/
TOKENIZER_MODEL=$HF_FORMAT_DIR
ITERATION=378000

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