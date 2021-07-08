export FAIRSEQ_PATH=../../../tools/fairseq

# stage 1. prepare data
mkdir -p data
python $FAIRSEQ_PATH/examples/wav2vec/wav2vec_manifest.py /DB/LibriSpeech/LibriSpeech \
    --dest data/train-960 \
    --ext flac \
    --valid-percent 0
python $FAIRSEQ_PATH/examples/wav2vec/wav2vec_manifest.py /DB/LibriSpeech/LibriSpeech-valid/dev-other \
    --dest data/dev-other \
    --ext flac \
    --valid-percent 0
cp data/dev-other/train.tsv data/train-960/valid.tsv

# stage 2. pre-training
## single-node
fairseq-hydra-train task.data=$PWD/data/train-960 distributed_training.distributed_world_size=4 optimization.update_freq='[43]'\
    --config-dir $FAIRSEQ_PATH/examples/wav2vec/config/pretraining \
    --config-name wav2vec2_large_librivox
### get checkpoints
# fairseq-hydra-train task.data=$PWD/data/train-960 checkpoint.save_dir=/home/jpong/Workspace/HYnet2-fairseq/egs/libirispeech/ssl1/outputs/2021-06-10/10-31-15/checkpoints distributed_training.distributed_world_size=4 optimization.update_freq='[43]'\
#     --config-dir $FAIRSEQ_PATH/examples/wav2vec/config/pretraining \
#     --config-name wav2vec2_large_librivox

# stage 3. fine-tunning
cd data
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt -O $PWD/data/train-clean-100/dict.ltr.txt
mkdir -p downloads $$ cd downloads
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt && cd ../
python $FAIRSEQ_PATH/examples/wav2vec/wav2vec_manifest.py /DB/LibriSpeech/LibriSpeech/train-clean-100 \
    --dest data/train-clean-100 \
    --ext flac \
    --valid-percent 0

python $FAIRSEQ_PATH/examples/wav2vec/libri_labels.py $PWD/data/train-clean-100/train.tsv \
    --output-dir $PWD/data/train-clean-100 \
    --output-name train
python $FAIRSEQ_PATH/examples/wav2vec/libri_labels.py $PWD/data/train-clean-100/valid.tsv \
    --output-dir $PWD/data/train-clean-100 \
    --output-name valid
cp data/dev-other/train.tsv data/train-clean-100/valid.tsv

# fairseq
fairseq-hydra-train task.data=$PWD/data/train-clean-100 \
    task.normalize=false model.w2v_path=$PWD/downloads/libri960_big.pt optimization.max_update=80000\
    dataset.valid_subset=valid \
    distributed_training.distributed_world_size=4 \
    optimization.update_freq='[8]' \
    model.freeze_finetune_updates=0 \
    --config-name vox_100h \
    --config-dir $FAIRSEQ_PATH/examples/wav2vec/config/finetuning

# stage 4. decoding
python $FAIRSEQ_PATH/examples/wav2vec/wav2vec_manifest.py /DB/LibriSpeech/LibriSpeech-valid/test-other \
    --dest data/test-other \
    --ext flac \
    --valid-percent 0
cp data/test-other/train.tsv data/train-clean-100/test.tsv
python $FAIRSEQ_PATH/examples/wav2vec/libri_labels.py $PWD/data/train-clean-100/test.tsv \
    --output-dir $PWD/data/train-clean-100 \
    --output-name test

python $FAIRSEQ_PATH/examples/speech_recognition/infer.py data/train-clean-100/ --task audio_pretraining \
--w2l-decoder viterbi --criterion ctc --labels ltr \
--post-process letter --path outputs/2021-06-13/20-03-09/checkpoints/checkpoint_best.pt