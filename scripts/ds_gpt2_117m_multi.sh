
source .env/bin/activate

# Dataset path & checkpoint path
DATASET_PATH=dataset/BookCorpusDataset_text_document
CHECKPOINT_PATH=checkpoints/gpt2_117m/
mkdir -p ${CHECKPOINT_PATH}

VOCAB_PATH=dataset/gpt2-vocab.json
MERGE_PATH=dataset/gpt2-merges.txt

# GPT-2 117M (12-layer, 768-hidden, 12-heads, 117M parameters)
NUM_LAYERS=12
HIDDEN_SIZE=768
NUM_ATTN_HEADS=12

# GPU resources
NUM_NODES=2
NUM_GPUS_PER_NODE=1
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))

# Parellel parameters
PP_SIZE=1
TP_SIZE=1

DP_SIZE=$((${NUM_GPUS} / (${PP_SIZE} * ${TP_SIZE})))

# Training parameters
GRAD_ACCUMULATION_STEPS=1

MICRO_BATCHSIZE=8
GLOBAL_BATCH_SIZE=$((MICRO_BATCHSIZE * DP_SIZE))

SEQ_LENGTH=1024
MAX_POSITION_EMBEDDINGS=1024

TRAINING_ITERATIONS=500000
SAVE_INTERVAL=10000
LR_DECAY_ITERATIONS=320000

LR=0.00015
LR_WARMUP_ITER=32000
SEED=1234

# deepspeed configuration
CONFIG_FILE=scripts/ds_config_gpt2_117m_${NUM_GPUS}.json
ZERO_STAGE=1

# for debug
export CUDA_LAUNCH_BLOCKING=1

# Run Command
deepspeed --force_multi --num_nodes ${NUM_NODES} \
  --num_gpus ${NUM_GPUS_PER_NODE} \
  --hostfile ./hostfile \
  pretrain_gpt.py \
  --tensor-model-parallel-size ${TP_SIZE} \
  --pipeline-model-parallel-size ${PP_SIZE} \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --num-attention-heads ${NUM_ATTN_HEADS} \
  --micro-batch-size ${MICRO_BATCHSIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --seq-length ${SEQ_LENGTH} \
  --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
  --train-iters ${TRAINING_ITERATIONS} \
  --save-interval ${SAVE_INTERVAL} \
  --lr-decay-iters ${LR_DECAY_ITERATIONS} \
  --data-path ${DATASET_PATH} \
  --vocab-file ${VOCAB_PATH} \
  --merge-file ${MERGE_PATH} \
  --data-impl mmap \
  --split 949,50,1 \
  --save ${CHECKPOINT_PATH} \
  --load ${CHECKPOINT_PATH} \
  --distributed-backend nccl \
  --override-lr-scheduler \
  --lr $LR \
  --lr-decay-style cosine \
  --min-lr 1.0e-5 \
  --weight-decay 1e-2 \
  --clip-grad 1.0 \
  --lr-warmup-iters $LR_WARMUP_ITER \
  --checkpoint-activations \
  --log-interval 100 \
  --eval-interval 100 \
  --eval-iters 10 \
  --fp16 \
  --seed $SEED \
  --no-masked-softmax-fusion \
  --deepspeed \
  --deepspeed_config ${CONFIG_FILE} \
  --zero-stage ${ZERO_STAGE} \
  --deepspeed-activation-checkpointing
