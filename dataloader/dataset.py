#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)

"""PyTorch compatible dataset modules."""

import os
import soundfile as sf
from torch.utils.data import Dataset
from dataloader.utils import find_files
import pickle
from PIL import Image
import numpy as np
import random

class SingleDataset(Dataset):
    def __init__(
        self,
        files,
        query="*.wav",
        load_fn=sf.read,
        return_utt_id=False,
        subset_num=-1,
    ):
        self.return_utt_id = return_utt_id
        self.load_fn = load_fn
        self.subset_num = subset_num
        self.files = files
        self.data_path = files[0]

        # print("files[1]  ",files[1])
        with open(files[1], 'rb') as f:
            self.pickle_list = pickle.load(f)



    def __getitem__(self, idx):
        # idx = int(idx%1000)
        data = self._data(idx)
            

        return data


    def __len__(self):
        # print("lenght ",len(self.pickle_list))
        return len(self.pickle_list)
    
    

    def _data(self, idx):
        # idx = int(idx%1000)
        # print("idx  ",idx)
        return self._load_data(self.pickle_list[idx], self.load_fn)
    

    def _load_data(self, filename, load_fn):
        data=[]


        reverb_path = os.path.join(self.data_path,filename["reverb_speech_path"].replace(".wav","_100.wav"))
        reverb_data, _ = load_fn(reverb_path, always_2d=True) # (T, C)
        # k=random.randrange(0,reverb_data.shape[0]-14400)
        k=0

        reverb_data = reverb_data[k:k+20000] #[k:k+16000] #
        
        
        # image_path = os.path.join(self.data_path,filename["view_image"])


        # color_image = np.array(Image.open(image_path))

        # speaker_location = filename["speaker_location"]
        # view_location = filename["view_location"]

        # speaker_view_location = np.array(speaker_location + view_location)
        # speaker_view_location = np.expand_dims(speaker_view_location, axis=1)
        
        rir_path = os.path.join(self.data_path,filename["mono_rir_path"])
        
        rir_data, _ = load_fn(rir_path, always_2d=True)        
           

        rir_data =  rir_data[0:4000]
        


        data = [reverb_data, rir_data]
        
        return data




