#!/bin/bash

mkdir data && cd data

# download the synthetic data
mkdir synthetic_dataset && cd synthetic_dataset
wget https://dl.fbaipublicfiles.com/large_objects/nvas/v16.zip
unzip v16.zip
cd ..

# download the librispeech data
wget https://dl.fbaipublicfiles.com/large_objects/nvas/LibriSpeech-wav.zip
unzip LibriSpeech-wav.zip

