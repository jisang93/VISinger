# Unofficial implementation of Korean VISinger

VISinger: Variational Inference with Adversarial Learning for End-to-End Singing Voice Snythesis [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9747664)]

## Overview
This repositroy contains PyTorch implementation of the Korean VISinger architecture, along with examples. Feel free to use/modify the code.

<p align="center">
    <img src=./assets/architecture.png width=40%>
    <p align="center"><em>Architecture of VISInger</em>
</p>

## Install Dependencies
```
## We tested on Linux/Ubuntu 20.04. 
## Install Python 3.8+ first (Anaconda recommended).

export PYTHONPATH=.
# build a virtual env (recommended).
conda create -n venv python=3.8
conda activate venv
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
sudo apt install -y sox libsox-fmt-mp3
```

## Training
### 1. Datasets
The supported datasets are
- [CSD](https://program.ismir2020.net/static/lbd/ISMIR2020-LBD-435-abstract.pdf): a single-singer Korean datasets contains 2.12 hours in total.

### 2. Preprocessing
Run base_preprocess.py for preprocessing.
```
python preprocessor/runs/base_preprocess.py --config config/datasets/svs/csd/preprocess.yaml
```
After that, run base_binarize.py for training.
```
python preprocessor/runs/base_binarize.py --config config/datasets/svs/csd/preprocess.yaml
```

### 3. Training
Trian model with
```
CUDA_VISIBLE_DEVICES=0 python tasks/runs/run.py --config config/models/visinger.yaml --exp_name "[dir]/[folder_name]"
```

### 4. Inference
You have to download the [pretrained models]() (will be uploaded) and put them in `./checkpoints/svs/visinger`. You have to prepare MIDI data which contains lyrics with the same amount of notes. We uploaded the sample file in `./data/source/svs/new_midi/` (will be uploaded).
You can inference new singing voice with
```
python inference/visinger.py
```
please setting the file path of MIDI data in `./inference/visinger.py`.

## Note
- Korean singing voice synthesis (SVS) do not requires duration prediction. We just split the each syallble into three components: `onset`, `nucleus`, and `coda`. SVS has a long vowel duration and the `nucleus` of Korean syllable is equivalent to the vowel. In this repository, we assigned `onset` and `coda` to a maximum three frames and assigned the remaining frames to the `nucleus`.
- We will upload the checkpoints of VISinger trained on CSD datasets (will be upload after march 2023)

## Acknowledgments
Our codes are influenced by the following repos:
- [NATSpeech](https://github.com/NATSpeech/NATSpeech)
- [HiFi-GAN](https://github.com/jik876/hifi-gan)
- [VITS](https://github.com/jaywalnut310/vits)
- [BigVGAN unofficial implementation](https://github.com/sh-lee-prml/BigVGAN)
