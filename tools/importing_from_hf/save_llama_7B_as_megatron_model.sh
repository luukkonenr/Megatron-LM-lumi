#!/bin/bash

TARGET_TP=4
TARGET_PP=1
MODEL_NAME_OR_PATH="/scratch/project_462000086/viking-v2/converted_models/viking_v2_7B_iter_0358000_bfloat16/" #HF model
MEGATRON_FORMAT_DIR="converted_checkpoint_v2_to_v3_iter_0358000_bfloat16"
TOKENIZER_MODEL=$MODEL_NAME_OR_PATH
ITERATION=1

python3 tools/checkpoint/util.py \
  --model-type GPT \
  --loader loader_llama2_hf \
  --saver megatron \
  --target-tensor-parallel-size ${TARGET_TP} \
  --target-pipeline-parallel-size ${TARGET_PP} \
  --load-dir ${MODEL_NAME_OR_PATH} \
  --save-dir ${MEGATRON_FORMAT_DIR} \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --iteration ${ITERATION}