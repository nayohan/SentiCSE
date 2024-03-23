#!/bin/bash

MODEL_NAME=$1
DATA_FILE=$2
TRAIN_EPOCH=$3
BATCH_SIZE=$4
LR=$5
MERTRIC=$6
EVAL_STEP=$7
TASK_NAME=$8
MLM_WEIGHT=${9}
POSIIVE_RATIO=${10}
INBATCH_RATIO=${11}
NEGATIVE_RATIO=${12}
HARD_NEG_RATION=${13}
DIAGONAL=${14}
PPNN=${15}
SAVE_NAME=${16}
GRAD_ACCM=${17}

DATA_FILE_DIR="data/$DATA_FILE"
OUTPUT_DIR="./result/$SAVE_NAME-$TASK_NAME-$MODEL_NAME-e$TRAIN_EPOCH-b$BATCH_SIZE-lr$LR"

# In this example, we show how to train SentiCSE using multiple GPU cards and PyTorch's distributed data parallel on MR dataset.
# Set how many GPUs to use
NUM_GPU=4
# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=96

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# python senticse.py \

python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID senticse.py \
    --model_name_or_path ${MODEL_NAME} \
    --train_file ${DATA_FILE_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs $(expr ${TRAIN_EPOCH}) \
    --per_device_train_batch_size $(expr ${BATCH_SIZE}) \
    --learning_rate $(expr ${LR}) \
    --max_seq_length 128 \
    --evaluation_strategy steps \
    --metric_for_best_model ${MERTRIC} \
    --load_best_model_at_end \
    --eval_steps $(expr ${EVAL_STEP}) \
    --gradient_accumulation_steps ${GRAD_ACCM} \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_mlm \
    --fp16 \
    --mlm_weight ${MLM_WEIGHT} \
    --positive_weight ${POSIIVE_RATIO} \
    --inbatch_weight ${INBATCH_RATIO} \
    --negative_weight ${NEGATIVE_RATIO} \
    --hard_neg_weight ${HARD_NEG_RATION} \
    --diagnoal ${DIAGONAL} \
    --ppnn ${PPNN} \
    "$@"
