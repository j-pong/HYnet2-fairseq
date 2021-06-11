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
fairseq-hydra-train task.data=$PWD/data/train-960 distributed_training.distributed_world_size=2 optimization.update_freq='[6]'\
    --config-dir $FAIRSEQ_PATH/examples/wav2vec/config/pretraining \
    --config-name wav2vec2_large_librivox

# stage 3. fine-tunning
## download dictionary
cd data
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt && cd ../
## download pre-training model
mkdir -p downloads $$ cd downloads
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt && cd ../

python $FAIRSEQ_PATH/examples/wav2vec/wav2vec_manifest.py /DB/LibriSpeech/LibriSpeech/train-clean-100 \
    --dest $data_path/train-clean-100 \
    --ext flac \
    --valid-percent 0
python libri_labels.py $PWD/data/train-clean-100 \
    --output-dir $PWD/data/train-clean-100 \
    --output-name train
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt -O $PWD/data/dict.ltr.txt

fairseq-hydra-train task.data=$PWD/data \
    task.normalize=true model.w2v_path=$PWD/downloads/wav2vec_vox_new.pt \
    distributed_training.distributed_world_size=4 \
    optimization.update_freq='[1]' \
    model.freeze_finetune_updates=100 \
    --config-name vox_100h \
    --config-dir $PWD/fairseq/examples/wav2vec/config/finetuning/