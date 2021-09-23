#!/bin/bash

TRAIN_PATH=~/Workspace/HYnet2-fairseq/egs/librispeech/ssl1
. $TRAIN_PATH/path.sh

train_mode=0    # 0 : pre-training / 1 : fine-tuning
checkpoint=0

# Multi-node config
nproc_per_node=1
nnodes=1
node_rank=0
master_addr="10.1.0.44"
master_port=6686

# Decoding config
decoding_mode=viterbi
model_path=$TRAIN_PATH/downloads/wav2vec_vox_100h_new.pt
lm_model=$TRAIN_PATH/data/lm_librispeech_word_transformer.pt
lexicon=$TRAIN_PATH/data/librispeech_lexicon.lst

if [ $train_mode == 0 ]; then
    if [ $checkpoint == 0 ]; then
        # multi-node
        NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=eth0 python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --nnodes=$nnodes --node_rank=$node_rank \
            --master_addr=$master_addr --master_port=$master_port --use_env \
            $(which fairseq-hydra-train) task.data=$TRAIN_PATH/data/train-960 optimization.update_freq='[43]' \
            --config-dir $FAIRSEQ_PATH/examples/wav2vec/config/pretraining \
            --config-name wav2vec2_large_librivox
    else
        # get checkpoints
        NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=eth0 python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --nnodes=$nnodes --node_rank=$node_rank \
            --master_addr=$master_addr --master_port=$master_port --use_env \
            $(which fairseq-hydra-train) task.data=$TRAIN_PATH/data/train-960 checkpoint.save_dir=$checkpoint optimization.update_freq='[43]' \
            --config-dir $FAIRSEQ_PATH/examples/wav2vec/config/pretraining \
            --config-name wav2vec2_large_librivox
    fi
else
    # multi-node
    NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=eth0 python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --nnodes=$nnodes --node_rank=$node_rank \
        --master_addr=$master_addr --master_port=$master_port --use_env \
        $(which fairseq-hydra-train) task.data=$TRAIN_PATH/data/train-clean-100 \
        task.normalize=false model.w2v_path=$TRAIN_PATH/downloads/libri960_big.pt optimization.max_update=80000 \
        dataset.valid_subset=valid \
        optimization.update_freq='[43]' \
        model.freeze_finetune_updates=0 \
        --config-name vox_100h \
        --config-dir $FAIRSEQ_PATH/examples/wav2vec/config/finetuning   
fi
