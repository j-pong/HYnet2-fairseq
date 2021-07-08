#!/usr/bin/env bash
# Set CUDA path 
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin

# Dependencies
sudo apt install libfftw3-dev libopenmpi-dev
sudo apt install build-essential cmake
sudo apt install libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev 
sudo apt install zlib1g-dev libbz2-dev liblzma-dev
sudo apt install pkg-config

# (Optional) Set up Anaconda and PyTorch
./setup_anaconda.sh anaconda base 3.8 
source anaconda/bin/activate 
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

pip3 install --upgrade pip
pip3 install soundfile editdistance packaging

# build fairseq
git clone https://github.com/pytorch/fairseq && cd fairseq
pip3 install --editable ./

# Install arrayfire
wget https://arrayfire.s3.amazonaws.com/3.8.0/ArrayFire-v3.8.0_Linux_x86_64.sh
./ArrayFire-v3.8.0_Linux_x86_64.sh

# Install flashlight
# Piz install mkl before excute above lines
git clone https://github.com/flashlight/flashlight.git && cd flashlight
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DFL_BACKEND=CUDA \
-DArrayFire_DIR=/home/jpong/Workspace/HYnet2-fairseq/tools/arrayfire/share/ArrayFire/cmake/
make -j$(nproc)
sudo make install

## flashlight binding
cd bindings/python
python3 setup.py install