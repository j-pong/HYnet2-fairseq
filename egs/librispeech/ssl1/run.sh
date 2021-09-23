#!/bin/bash

TRAIN_PATH=~/Workspace/HYnet2-fairseq/egs/librispeech/ssl1
. $TRAIN_PATH/path.sh

stage=2
train_mode=1    # 0 : pre-training / 1 : fine-tuning
checkpoint=0

# Decoding config
decoding_mode=viterbi
model_path=$TRAIN_PATH/wav2vec_small_100h.pt
lm_model=$TRAIN_PATH/data/lm_librispeech_word_transformer.pt
lexicon=$TRAIN_PATH/data/librispeech_lexicon.lst

# stage 0. prepare data
if [ $stage -le 0 ]; then
    mkdir -p data
    python $FAIRSEQ_PATH/examples/wav2vec/wav2vec_manifest.py /DB/LibriSpeech/LibriSpeech \
        --dest $TRAIN_PATH/data/train-960 \
        --ext flac \
        --valid-percent 0
    python $FAIRSEQ_PATH/examples/wav2vec/wav2vec_manifest.py /DB/LibriSpeech/LibriSpeech-valid/dev-other \
        --dest $TRAIN_PATH/data/dev-other \
        --ext flac \
        --valid-percent 0
    cp $TRAIN_PATH/data/dev-other/train.tsv $TRAIN_PATH/data/train-clean-100/valid.tsv
    cp $TRAIN_PATH/data/train-960/train.tsv $TRAIN_PATH/data/train-clean-100/train.tsv

    python $FAIRSEQ_PATH/examples/wav2vec/libri_labels.py $TRAIN_PATH/data/train-clean-100/train.tsv \
        --output-dir $TRAIN_PATH/data/train-clean-100 \
        --output-name train
    python $FAIRSEQ_PATH/examples/wav2vec/libri_labels.py $TRAIN_PATH/data/train-clean-100/valid.tsv \
        --output-dir $TRAIN_PATH/data/train-clean-100 \
        --output-name valid

fi

# stage 1. pre-training
if [ $stage -le 1 ]; then
    # pre-training
    if [ $train_mode == 0 ]; then
        if [ -z $checkpoint ]; then
            fairseq-hydra-train task.data=$TRAIN_PATH/data/train-960 distributed_training.distributed_world_size=4 optimization.update_freq='[43]' \
                --config-dir $FAIRSEQ_PATH/examples/wav2vec/config/pretraining \
                --config-name wav2vec2_large_librivox
        else
            # get checkpoints
            fairseq-hydra-train task.data=$TRAIN_PATH/data/train-960 checkpoint.save_dir=$checkpoint distributed_training.distributed_world_size=4 optimization.update_freq='[43]' \
                --config-dir $FAIRSEQ_PATH/examples/wav2vec/config/pretraining \
                --config-name wav2vec2_large_librivox
        fi
    # fine-tunning
    else
        if [ ! -e $TRAIN_PATH/data/train-clean-100/dict.ltr.txt ]; then
            wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt -O $TRAIN_PATH/data/train-clean-100/dict.ltr.txt
        fi

        if [ ! -e $TRAIN_PATH/downloads/libri960_big.pt ]; then
            mkdir -p downloads
            wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt -O $TRAIN_PATH/downloads/libri960_big.pt
        fi

        python $FAIRSEQ_PATH/examples/wav2vec/wav2vec_manifest.py /DB/LibriSpeech/LibriSpeech/train-clean-100 \
            --dest $TRAIN_PATH/data/train-clean-100 \
            --ext flac \
            --valid-percent 0
            
        fairseq-hydra-train task.data=$TRAIN_PATH/data/train-clean-100 \
        task.normalize=false model.w2v_path=$TRAIN_PATH/downloads/libri960_big.pt optimization.max_update=80000 \
        dataset.valid_subset=valid \
        optimization.update_freq='[8]' \
        model.freeze_finetune_updates=0 \
        --config-name vox_100h \
        --config-dir $FAIRSEQ_PATH/examples/wav2vec/config/finetuning
    fi
fi

# stage 2. Decoding
if [ $stage -le 2 ]; then
    if [ ! -e $TRAIN_PATH/data/train-clean-100/test.tsv ]; then
        python $FAIRSEQ_PATH/examples/wav2vec/wav2vec_manifest.py /DB/LibriSpeech/LibriSpeech-valid/test-other \
            --dest $TRAIN_PATH/data/test-other \
            --ext flac \
            --valid-percent 0
        cp data/test-other/train.tsv $TRAIN_PATH/data/train-clean-100/test.tsv

        python $FAIRSEQ_PATH/examples/wav2vec/libri_labels.py $TRAIN_PATH/data/train-clean-100/test.tsv \
            --output-dir $TRAIN_PATH/data/train-clean-100 \
            --output-name test
    fi
    if [ $decoding_mode == viterbi ]; then
        # viterbi decoding
        python $FAIRSEQ_PATH/examples/speech_recognition/infer.py $TRAIN_PATH/data/train-clean-100/ --task audio_pretraining \
        --w2l-decoder viterbi --criterion ctc --labels ltr \
        --post-process letter --path $model_path
    else
        # transformer LM decoding
        python $FAIRSEQ_PATH/examples/speech_recognition/infer.py $TRAIN_PATH/data/train-clean-100/ --task audio_pretraining \
        --nbest 1 --path $model_path --results-path outputs --w2l-decoder fairseqlm --lm-model $lm_model \
        --lm-weight 1 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 1280000 --lexicon $lexicon --remove-bpe letter
    fi
fi