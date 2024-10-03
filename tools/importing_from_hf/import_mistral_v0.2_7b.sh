#!/bin/bash

TARGET_PP=1
TARGET_TP=1
# MODEL_NAME_OR_PATH="mistral-community/Mistral-7B-v0.2" #HF model
# MODEL_NAME_OR_PATH="/scratch/project_462000319/jburdge/cache/huggingface/hub/models--mistral-community--Mistral-7B-v0.2/snapshots/2c3e624962b1a3f3fbf52e15969565caa7bc064a/"
MODEL_NAME_OR_PATH="/scratch/project_462000353/risto/Megatron-LM-lumi-dev/FIN-ENG-CODE-Extended-Mistral-7B-v0.2"
MEGATRON_FORMAT_DIR="imported_checkpoints/Mistral-7B-v0.2-megatron"
TOKENIZER_MODEL=$MODEL_NAME_OR_PATH
ITERATION=1

python3 tools/checkpoint/util.py \
  --model-type GPT \
  --loader loader_mistral_v02_hf \
  --saver megatron \
  --target-tensor-parallel-size ${TARGET_TP} \
  --target-pipeline-parallel-size ${TARGET_PP} \
  --load-dir ${MODEL_NAME_OR_PATH} \
  --save-dir ${MEGATRON_FORMAT_DIR} \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --iteration ${ITERATION}