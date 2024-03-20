#!/bin/bash

#SBATCH --job-name=p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH -p dev-g
#SBATCH -t 00-00:15:00
#SBATCH --gpus-per-node=mi250:8
#SBATCH --hint=nomultithread
#SBATCH --exclusive=user
#SBATCH --account=project_462000319
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --cpus-per-task=7

mkdir -p workdir
wd=$(realpath workdir)


# if run without sbatch, invoke here
if [ -z $SLURM_JOB_ID ]; then
    mkdir -p logs
    sbatch "$0"
    exit
fi

# distributed setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS

# compilers in the container
export CC=gcc-10
export CXX=g++-10

# singularity setup

# CONTAINER="/pfs/lustrep4/scratch/project_462000319/rluukkon/singularity/flash-attn-test/"
CONTAINER=/scratch/project_462000319/containers/vaino_flashattention_v2_new/
SING_BIND="/scratch/project_462000319,/flash/project_462000319"

#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib"

# hold separate logs for easier debugging
rm -rf separate-logs
mkdir -p separate-logs


set -euo pipefail

# symlink logs/latest.out and logs/latest.err
ln -f -s "$SLURM_JOB_ID.out" logs/latest.out
ln -f -s "$SLURM_JOB_ID.err" logs/latest.err

UNIQ_MODEL_NAME="megatron-sp-large"
CHECKPOINT_PATH="checkpoints/$UNIQ_MODEL_NAME"_data4t_test
TENSORBOARD_PATH="tensorboard/$UNIQ_MODEL_NAME/$SLURM_JOB_ID"
#rm -rf "$CHECKPOINT_PATH" "$TENSORBOARD_PATH" # Start from scratch

# Data

LEARNING_RATE=2.5e-4
export CUDA_DEVICE_MAX_CONNECTIONS=1

EVAL_INTERVAL=500
EVAL_STEPS=100

# DATA_PATH="0.07370182629 /scratch/project_462000319/viking_preprocessed_data/merged_datasets/finnish 0.3302641761 /scratch/project_462000319/viking_preprocessed_data/merged_datasets/slimpajama 0.330497442 /scratch/project_462000319/viking_preprocessed_data/merged_datasets/starcoderdata 0.08367352788 /scratch/project_462000319/viking_preprocessed_data/merged_datasets/nordic-en-xling-combined 0.002361170146 /scratch/project_462000319/viking_preprocessed_data/small_files/train-books_text_document 0.05157063372 /scratch/project_462000319/viking_preprocessed_data/nordics/mc4-da-train_text_document 0.004054463623 /scratch/project_462000319/viking_preprocessed_data/nordics/mc4-is-train_text_document 0.08052558051 /scratch/project_462000319/viking_preprocessed_data/nordics/mc4-sv-train_text_document 0.04188033719 /scratch/project_462000319/viking_preprocessed_data/nordics/nor_all_combined_text_document 0.001470842506 /scratch/project_462000319/viking_preprocessed_data/small_files/natural_instruct_train_text_document"
# DATA_PATH="1.0 /scratch/project_462000319/viking_preprocessed_data/small_files/natural_instruct_train_text_document"
# DATA_PATH="1.0 mock-data_text_document"
MERGES=/scratch/project_462000319/tokenizers/nordic_tokenizer_131072/merges.txt
VOCAB=/scratch/project_462000319/tokenizers/nordic_tokenizer_131072/vocab.json
TRAIN_DATA="0.5415810341 /flash/project_462000319/megatron-preprocessed-data/train/merged_slimpajama 0.1304808053 /flash/project_462000319/megatron-preprocessed-data/train/merged_finnish 0.004023063515 /flash/project_462000319/megatron-preprocessed-data/train/tatoeba-train.en-fi.jsonl_text_document 0.004016818638 /flash/project_462000319/megatron-preprocessed-data/train/tatoeba-train.fi-en.jsonl_text_document 0.3153543717 /flash/project_462000319/megatron-preprocessed-data/train/starcoder-merged 0.004543906834 /flash/project_462000319/megatron-preprocessed-data/train/train-books_text_document"
VALIDATION_DATA="0.5415810341 /flash/project_462000319/megatron-preprocessed-data/eval/slim-pajama-validation_text_document 0.1304808053 /flash/project_462000319/megatron-preprocessed-data/eval/finnish_eval_text_document 0.004023063515 /flash/project_462000319/megatron-preprocessed-data/eval/tatoeba-eval.en-fi_text_document 0.004016818638 /flash/project_462000319/megatron-preprocessed-data/eval/tatoeba-eval.fi-en_text_document 0.3153543717 /flash/project_462000319/megatron-preprocessed-data/eval/starcoder-eval_content_document 0.004543906834 /flash/project_462000319/megatron-preprocessed-data/eval/eval-books.json_text_document"

# DATA_PATH="1.0 /scratch/project_462000319/viking_preprocessed_data/merged_datasets/finnish"
# DATA_PATH="0.5415810341 /flash/project_462000319/megatron-preprocessed-data/train/merged_slimpajama 0.1304808053 /flash/project_462000319/megatron-preprocessed-data/train/merged_finnish 0.004023063515 /flash/project_462000319/megatron-preprocessed-data/train/tatoeba-train.en-fi.jsonl_text_document 0.004016818638 /flash/project_462000319/megatron-preprocessed-data/train/tatoeba-train.fi-en.jsonl_text_document 0.3153543717 /flash/project_462000319/megatron-preprocessed-data/train/starcoder-merged 0.004543906834 /flash/project_462000319/megatron-preprocessed-data/train/train-books_text_document"
PP_SIZE=1
TP_SIZE=1

MICRO_BATCH_SIZE=1
# GBS_IN_TOKENS=4194304
SEQ_LEN=5120
# GLOBAL_BATCH_SIZE=$((GBS_IN_TOKENS/SEQ_LEN))
GLOBAL_BATCH_SIZE=32
# export MEMORY_OPT_ALLREDUCE_SIZE=2500000000
export MEMORY_OPT_ALLREDUCE_SIZE=1500000000
echo "MEMORY_OPT_ALLREDUCE_SIZE $MEMORY_OPT_ALLREDUCE_SIZE"

# TRAIN_SAMPLES=5000000
# TOTAL_TOKENS=4_000_000_000 # 4 trillion
TOTAL_TOKENS=3_000_000_000 # 3 trillion
TOTAL_TOKENS=${TOTAL_TOKENS//_}    # drop "_" for bash math
TRAIN_SAMPLES=$((TOTAL_TOKENS/SEQ_LEN))
LR_DECAY_SAMPLES=$TRAIN_SAMPLES
LR_WARMUP_SAMPLES=$((TRAIN_SAMPLES/100))
echo "total tokens $TOTAL_TOKENS, train samples $TRAIN_SAMPLES"


MICRO_BATCH_SIZE=1

NHIDDEN=1024
FFN_HIDDEN_SIZE=$((4*NHIDDEN))
NLAYERS=24
NHEADS=16


SAVE_INTERVAL=50


EVAL_INTERVAL=60000
EVAL_STEPS=100

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr $LEARNING_RATE \
    --min-lr 2e-5 \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "

GPT_ARGS=" \
    --use-distributed-optimizer \
    --no-gradient-accumulation-fusion \
    --no-async-tensor-model-parallel-allreduce \
    --untie-embeddings-and-output-weights \
    --no-masked-softmax-fusion \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file $VOCAB \
    --merge-file $MERGES \
    --init-method-std 0.015 \
    --use-flash-attn \
    --bf16 \
    --seed 42 \
    --swiglu \
    --no-query-key-layer-scaling \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --num-workers 2 \
    --disable-bias-linear \
    --make-vocab-size-divisible-by 128 \
    --normalization RMSNorm \
    --no-gradient-accumulation-fusion \
    --untie-embeddings-and-output-weights \
    --sequence-parallel \
    --use-rotary-position-embeddings \
    $OPTIMIZER_ARGS \
    "
    # --num-experts 8 \
    # --expert-parallel \
    # --num-key-value-heads 8 \
    # --no-masked-softmax-fusion \
    # --no-position-embedding \
    # --no-async-ensor-model-parallel-allreduce \
    # --no-pipeline-parallel \
    # --deepspeed-activation-checkpointing \
    # --checkpoint-activations \
    # --sync-tp-duplicated-parameters \
    # --embed-layernorm \

OUTPUT_ARGS=" \
    --log-interval 10 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters $EVAL_STEPS \
    --tensorboard-dir $TENSORBOARD_PATH \
    --no-log-loss-scale-to-tensorboard \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "
    # --load $CHECKPOINT_PATH \
    # --save $CHECKPOINT_PATH \
    

CMD=" \
    pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --train-data-path $VALIDATION_DATA \
    --valid-data-path $TRAIN_DATA \
    --dataloader-type single \
    "
    # --exit-interval 200 \
    # --profile

echo $CMD


c="fe"

# Bind mask for one thread per core
BIND_MASK_1="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

# # Bind mask for two threads per core
# BIND_MASK_2="0x${c}00000000000000${c}000000000000,0x${c}00000000000000${c}00000000000000,0x${c}00000000000000${c}0000,0x${c}00000000000000${c}000000,0x${c}00000000000000${c},0x${c}00000000000000${c}00,0x${c}00000000000000${c}00000000,0x${c}00000000000000${c}0000000000"

BIND_MASK="$BIND_MASK_1"
echo "Using --cpu-bind=mask_cpu:$BIND_MASK"

echo "START $SLURM_JOBID: $(date)"

if [ ! -d $wd/cray-deps ] ; then
  rm -rf $wd/cray-deps
  mkdir $wd/cray-deps
  cp /usr/lib64/libcxi* $wd/cray-deps
fi

    # --cpu-bind=mask_cpu:$BIND_MASK \
srun \
    --label \
    singularity exec \
    -B /opt/cray:/opt/cray \
    -B "$wd/cray-deps":/opt/cray-deps \
    -B "$wd":/workdir \
    -B "$SING_BIND" \
    "$CONTAINER" \
    ./launch.sh \
    $CMD

echo "END $SLURM_JOBID: $(date)"
