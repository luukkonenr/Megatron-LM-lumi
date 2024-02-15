#!/bin/bash

#SBATCH --exclude=nid006865,nid005613,nid005988,nid006657,nid[005325-005331],nid006611,,nid006198,nid005120,nid005122,nid005250,nid005469
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --partition=dev-g
#SBATCH --time=0-00:30:00
#SBATCH --gpus-per-node=mi250:8
##SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --job-name=debug_scale
#SBATCH --account=project_462000353
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

mkdir -p workdir
wd=$(realpath workdir)


# if run without sbatch, invoke here
#if [ -z $SLURM_JOB_ID ]; then
#    mkdir -p logs
#    sbatch "$0"
#    exit
#fi

# distributed setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS

# compilers in the container
export CC=gcc-10
export CXX=g++-10

# singularity setup

# CONTAINER="/projappl/project_462000007/hatanpav/singularity/flashattention_new"
CONTAINER="/scratch/project_462000319/containers/vaino_flashattention_v2_new/"
mkdir -p pythonuserbase
export PYTHONUSERBASE=pythonuserbase
# SING_BIND="/flash/project_462000424/sp_debug_data,/scratch/project_462000007/hatanpav,/scratch/project_462000319/tokenizers/tokenizer_v6_fixed_fin/,/projappl/project_462000007/hatanpav,/flash/project_462000007/hatanpav/data/LLM-throughput/,/scratch/project_462000424/sp_debug_data"
SING_BIND="/scratch/project_462000319,/flash/project_462000319,/scratch/project_462000086"

#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib"

# hold separate logs for easier debugging
rm -rf separate-logs
mkdir -p separate-logs

LEARNING_RATE=1.5e-4    # TODO probably too low

set -euo pipefail

# symlink logs/latest.out and logs/latest.err
ln -f -s "$SLURM_JOB_ID.out" logs/latest.out
ln -f -s "$SLURM_JOB_ID.err" logs/latest.err

# CHECKPOINT_PATH=/scratch/project_462000007/hatanpav/output/LumiLLM/checkpoints_v3_megsp
# TENSORBOARD_PATH="tensorboard/$SLURM_JOB_ID"

CHECKPOINT_PATH=checkpoints
TENSORBOARD_PATH="tensorboard/7B_high_eps.$SLURM_JOB_ID"
#rm -rf "$CHECKPOINT_PATH" "$TENSORBOARD_PATH" # Start from scratch

# Data

export CUDA_DEVICE_MAX_CONNECTIONS=1

#DATA_PATH=/projappl/project_462000007/hatanpav/debugging/LLM-throughput/Singularity/lumi-llm-scaling/meg-ds-sing/data/wikipedia_20220301.en.train_text_document
#DATA_PATH=/flash/project_462000007/hatanpav/data/LLM-throughput/input_data/wikipedia_20220301.en.train_text_document
#DATA_PATH=/projappl/project_462000007/hatanpav/debugging/LLM-throughput/NV/Megatron-LM/my-gpt2_text_document
# DATA_PATH="0.5415810341 /scratch/project_462000424/sp_debug_data/merged_slimpajama 0.1304808053 /scratch/project_462000424/sp_debug_data/merged_finnish 0.004023063515 /scratch/project_462000424/sp_debug_data/tatoeba-train.en-fi.jsonl_text_document 0.004016818638 /scratch/project_462000424/sp_debug_data/tatoeba-train.fi-en.jsonl_text_document 0.3153543717 /scratch/project_462000424/sp_debug_data/starcoder-merged 0.004543906834 /scratch/project_462000424/sp_debug_data/train-books_text_document"
DATA_PATH="1.0 /scratch/project_462000319/rluukkon/Megatron-DeepSpeed-dev/dataset/parsebank-combined.dedup.filtered.jsonl-with-reg-scores-MT-filtered_text_document"


MERGES=/scratch/project_462000319/tokenizers/nordic_tokenizer_131072/merges.txt
VOCAB=/scratch/project_462000319/tokenizers/nordic_tokenizer_131072/vocab.json

PP_SIZE=8
TP_SIZE=8

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024
GLOBAL_BATCH_SIZE=$((1024/64))

NLAYERS=96
NHIDDEN=10240
NHEADS=80
FFN_HIDDEN_SIZE=30720
SEQ_LEN=4096

LOG_INTERVAL=1
SAVE_INTERVAL=1000
EVAL_INTERVAL=500
EVAL_STEPS=100

TOTAL_TOKENS=2_000_000_000 # 2 trillion
TOTAL_TOKENS=${TOTAL_TOKENS//_}    # drop "_" for bash math
TRAIN_SAMPLES=$((TOTAL_TOKENS/SEQ_LEN))
LR_DECAY_SAMPLES=$TRAIN_SAMPLES
# LR_WARMUP_SAMPLES=$((GLOBAL_BATCH_SIZE*2000))
LR_WARMUP_SAMPLES=$((TRAIN_SAMPLES/100))
# LR_WARMUP_SAMPLES=$((GLOBAL_BATCH_SIZR_WARMUP_SAMPLES=$((TRAIN_SAMPLES/100)) E*2000))

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --lr $LEARNING_RATE \
    --min-lr 2e-5 \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "

GPT_ARGS=" \
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
    --init-method-std 0.0048 \
    --vocab-file $VOCAB \
    --merge-file $MERGES \
    --bf16 \
    --seed 42 \
    --make-vocab-size-divisible-by 128 \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --no-query-key-layer-scaling \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --use-flash-attn \
    --group-query-attention \
    --num-query-groups 8 \
    --normalization RMSNorm \
    --num-layers-per-virtual-pipeline-stage 3 \
    --use-rotary-position-embeddings \
    --use-distributed-optimizer \
    --no-bias-dropout-fusion \
    --no-gradient-accumulation-fusion \
    $OPTIMIZER_ARGS \
    "

    #--use-distributed-optimizer \
    #--normalization rmsnorm \

    #--num-layers-per-virtual-pipeline-stage 1 \
    #--recompute-granularity full \
    #--recompute-method uniform \
    #--recompute-num-layers 2 \
    # --embed-layernorm \
    # --sync-tp-duplicated-parameters \
    # --position-embedding-type alibi \

OUTPUT_ARGS=" \
    --log-interval $LOG_INTERVAL \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters $EVAL_STEPS \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "
    # --tensorboard-dir $TENSORBOARD_PATH \
    # --tensorboard-queue-size 5 \
    # --log-timers-to-tensorboard \
    # --log-batch-size-to-tensorboard \
    # --log-validation-ppl-to-tensorboard \

CMD=" \
    pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --sequence-parallel \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --data-path $DATA_PATH \
    --dataloader-type single \
    --num-workers 0 \
    --profile \
    --profile-step-end 20 \
    "
    # --save $CHECKPOINT_PATH \

export TORCH_EXTENSIONS_DIR=torch_extensions/$SLURM_JOBID
mkdir -p $TORCH_EXTENSIONS_DIR

echo $CMD

echo "START $SLURM_JOBID: $(date)"

srun \
    --label \
    --cpus-per-task=7 \
    singularity exec \
    -B /var/spool/slurmd \
    -B /opt/cray \
    -B /usr/lib64/libcxi.so.1 \
    -B /usr/lib64/libjansson.so.4 \
    -B "$SING_BIND" \
    "$CONTAINER" \
    ./launch.sh \
    $CMD

echo "END $SLURM_JOBID: $(date)"