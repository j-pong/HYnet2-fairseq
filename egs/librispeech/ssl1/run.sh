export FAIRSEQ_PATH=../../../tools/fairseq

# stage 1. prepare data
    mkdir -p data
    ## pre-training dataset
    python $FAIRSEQ_PATH/examples/wav2vec/wav2vec_manifest.py /DB/LibriSpeech/LibriSpeech \
        --dest data/train-960 \
        --ext flac \
        --valid-percent 0

    ## unlabeled dataset
    python $FAIRSEQ_PATH/examples/wav2vec/wav2vec_manifest.py /DB/LibriSpeech/LibriSpeech/train-860 \
        --dest data/train-860 \
        --ext flac \
        --valid-percent 0

    ## labeled dataset
    python $FAIRSEQ_PATH/examples/wav2vec/wav2vec_manifest.py /DB/LibriSpeech/LibriSpeech/train-clean-100 \
        --dest data/train-clean-100 \
        --ext flac \
        --valid-percent 0

    ## validation dataset
    python $FAIRSEQ_PATH/examples/wav2vec/wav2vec_manifest.py /DB/LibriSpeech/LibriSpeech-valid/dev-other \
        --dest data/dev-other \
        --ext flac \
        --valid-percent 0
    python $FAIRSEQ_PATH/examples/wav2vec/wav2vec_manifest.py /DB/LibriSpeech/LibriSpeech-valid/dev-clean \
        --dest data/dev-clean \
        --ext flac \
        --valid-percent 0
    python $FAIRSEQ_PATH/examples/wav2vec/wav2vec_manifest.py /DB/LibriSpeech/LibriSpeech-valid/test-clean \
        --dest data/test-clean \
        --ext flac \
        --valid-percent 0
    python $FAIRSEQ_PATH/examples/wav2vec/wav2vec_manifest.py /DB/LibriSpeech/LibriSpeech-valid/test-other \
        --dest data/test-other \
        --ext flac \
        --valid-percent 0

# stage 2. pre-training
## if pre-traning:
    cp data/dev-other/train.tsv data/train-960/valid.tsv

    fairseq-hydra-train task.data=$PWD/data/train-960 distributed_training.distributed_world_size=4 optimization.update_freq='[43]'\
        --config-dir $FAIRSEQ_PATH/examples/wav2vec/config/pretraining \
        --config-name wav2vec2_large_librivox
    ### get checkpoints
    # fairseq-hydra-train task.data=$PWD/data/train-960 checkpoint.save_dir=/home/jpong/Workspace/HYnet2-fairseq/egs/libirispeech/ssl1/outputs/2021-06-10/10-31-15/checkpoints distributed_training.distributed_world_size=4 optimization.update_freq='[43]'\
    #     --config-dir $FAIRSEQ_PATH/examples/wav2vec/config/pretraining \
    #     --config-name wav2vec2_large_librivox

## else:
    mkdir -p downloads $$ cd downloads
    wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt && cd ../

# stage 3. fine-tunning
    cp data/dev-other/train.tsv data/train-clean-100/valid.tsv

    # get transcription
    wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt -O $PWD/data/train-clean-100/dict.ltr.txt
    python $FAIRSEQ_PATH/examples/wav2vec/libri_labels.py $PWD/data/train-clean-100/train.tsv \
        --output-dir $PWD/data/train-clean-100 \
        --output-name train
    python $FAIRSEQ_PATH/examples/wav2vec/libri_labels.py $PWD/data/train-clean-100/valid.tsv \
        --output-dir $PWD/data/train-clean-100 \
        --output-name valid

    fairseq-hydra-train task.data=$PWD/data/train-clean-100 \
        task.normalize=false model.w2v_path=$PWD/downloads/libri960_big.pt optimization.max_update=80000\
        dataset.valid_subset=valid \
        distributed_training.distributed_world_size=4 \
        optimization.update_freq='[8]' \
        model.freeze_finetune_updates=0 \
        --config-name vox_100h \
        --config-dir $FAIRSEQ_PATH/examples/wav2vec/config/finetuning

# stage 4. decoding
    cp data/test-clean/train.tsv data/train-clean-100/test.tsv
    python $FAIRSEQ_PATH/examples/wav2vec/libri_labels.py $PWD/data/train-clean-100/test.tsv \
        --output-dir $PWD/data/train-clean-100 \
        --output-name test

    python $FAIRSEQ_PATH/examples/speech_recognition/infer.py data/train-clean-100/ --task audio_pretraining \
    --w2l-decoder viterbi --criterion ctc --labels ltr \
    --post-process letter --path outputs/finetunning/2021-07-03/epoch8000/checkpoints/checkpoint_best.pt

# stage 5. generate pseudo label
mkdir data/train-nl-860
cp data/train-860/train.tsv data/train-nl-860/test.tsv
python $FAIRSEQ_PATH/examples/wav2vec/libri_labels.py $PWD/data/train-nl-860/test.tsv \
    --output-dir $PWD/data/train-nl-860 \
    --output-name test

python $FAIRSEQ_PATH/examples/speech_recognition/infer.py data/train-nl-860 \
    --task audio_pretraining --w2l-decoder viterbi --criterion ctc \
    --labels ltr --post-process letter --quiet \
    --path outputs/finetunning/2021-07-03/epoch8000/checkpoints/checkpoint_best.pt \
    --results-path outputs/finetunning/2021-07-03/epoch8000/train-nl-860/ 

python local/parse_pseudo_label.py $PWD/data/train-nl-860/train.tsv \
    --output-dir $PWD/data/train-nl-860 \
    --output-name train \
    --pseudo-label-dir $PWD/outputs/finetunning/2021-07-03/epoch8000/train-nl-860/hypo.word-checkpoint_best.pt-test.txt

# stage 6. self-training
# Note. merge train-nl-860 and train-clean-100 before running below lines
cp data/dev-other/train.tsv data/train-nl-960/valid.tsv
python $FAIRSEQ_PATH/examples/wav2vec/libri_labels.py $PWD/data/train-nl-960/valid.tsv \
        --output-dir $PWD/data/train-nl-960 \
        --output-name valid

fairseq-hydra-train task.data=$PWD/data/train-nl-960 \
        task.normalize=false model.w2v_path=$PWD/downloads/libri960_big.pt optimization.max_update=80000\
        dataset.valid_subset=valid \
        distributed_training.distributed_world_size=4 \
        optimization.update_freq='[8]' \
        model.freeze_finetune_updates=0 \
        --config-name vox_100h \
        --config-dir $FAIRSEQ_PATH/examples/wav2vec/config/finetuning
