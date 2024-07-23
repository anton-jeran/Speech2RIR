#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)

import os
import logging
import argparse
import torch
import soundfile as sf
import cupy as cp
import numpy as np
import scipy.signal
from torch.utils.data import DataLoader
from dataloader import CollaterAudio, CollaterAudioPair,CollaterAudioSet
from dataloader import SingleDataset #, MultiDataset

from models.autoencoder.AudioDec import Generator as generator_audiodec
from models.HiFiGAN_Generator import Generator as generator_hifigan

from models.vocoder.FASTRIR import Discriminator_RIR as discriminator_rir

from trainer.autoencoder import Trainer as TrainerAutoEncoder
from trainer.vocoder import Trainer as TrainerVocoder
# from trainer.denoise import Trainer as TrainerDenoise
from bin.train import TrainGAN

from losses import DiscriminatorAdversarialLoss
from losses import FeatureMatchLoss
from losses import GeneratorAdversarialLoss

from losses import MultiResolutionSTFTLoss
# from losses import MultiMelSpectrogramLoss
# from losses import MultiWindowShapeLoss
from losses import TimeDomainMSELoss
from losses import EnergyDecayCurveLoss
from losses import STOILoss


def generate_complementary_filterbank(
    fc=[125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0],
    fs=48000,
    filter_order=4,
    filter_length=16384,
    power=True):
    """Return a zero-phase power (or amplitude) complementary filterbank via Butterworth prototypes.
    Parameters:
        fc - filter center frequencies
        fs - sampling rate
        filter_order - order of the prototype Butterworth filters
        filter_length - length of the resulting zero-phase FIR filters
        power - boolean to set if the filter is power or amplitude complementary
    """

    # sort in increasing cutoff
    fc = np.sort(fc)

    assert fc[-1] <= fs/2

    numFilts = len(fc)
    nbins = filter_length
    signal_z1 = np.zeros(2 * nbins)
    signal_z1[0] = 1
    irBands = np.zeros((2 * nbins, numFilts))

    for i in range(numFilts - 1):
        wc = fc[i] / (fs/2.0)
        # if wc >= 1:
        #     wc = .999999

        B_low, A_low = scipy.signal.butter(filter_order, wc, btype='low')
        B_high, A_high = scipy.signal.butter(filter_order, wc, btype='high')


        # Store the low band
        irBands[:, i] = scipy.signal.lfilter(B_low, A_low, signal_z1)

        # Store the high
        signal_z1 = scipy.signal.lfilter(B_high, A_high, signal_z1)

        # Repeat for the last band of the filter bank
    irBands[:, -1] = signal_z1

    # Compute power complementary filters
    if power:
        ir2Bands = np.real(np.fft.ifft(np.square(np.abs(np.fft.fft(irBands, axis=0))), axis=0))
    else:
        ir2Bands = np.real(np.fft.ifft(np.abs(np.abs(np.fft.fft(irBands, axis=0))), axis=0))

    ir2Bands = np.concatenate((ir2Bands[nbins:(2 * nbins), :], ir2Bands[0:nbins, :]), axis=0)

    return ir2Bands
class TrainMain(TrainGAN):
    def __init__(self, args,):
        super(TrainMain, self).__init__(args=args,)
        self.train_mode = self.config.get('train_mode', 'autoencoder')
        self.model_type = self.config.get('model_type', 'symAudioDec')
        self.data_path = self.config['data']['data_path']


    # DATA LOADER
    def initialize_data_loader(self):
        logging.info(f"Loading datasets... (batch_lenght: {self.batch_length})")

        if self.train_mode in ['autoencoder', 'vocoder']:
            train_set = self._audio_set('train')
            valid_set = self._audio_set('valid')
            collater = CollaterAudioSet(batch_length=self.batch_length)
        
        else:
            raise NotImplementedError(f"Train mode: {self.train_mode} is not supported!")

        logging.info(f"The number of training files = {len(train_set)}.")
        logging.info(f"The number of validation files = {len(valid_set)}.")
        dataset = {'train': train_set, 'dev': valid_set}
        self._data_loader(dataset, collater)
    

    def _data_loader(self, dataset, collater):
        self.data_loader = {
            'train': DataLoader(
                dataset=dataset['train'],
                shuffle=True,
                collate_fn=collater,
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                pin_memory=self.config['pin_memory'],
            ),
            'dev': DataLoader(
                dataset=dataset['dev'],
                shuffle=False,
                collate_fn=collater,
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                pin_memory=self.config['pin_memory'],
            ),
        }
    
    
  

    def _audio_set(self, subset, subset_num=-1, return_utt_id=False):
        audio_pickle_file = self.config['data']['subset'][subset]
        params = {
            'files': [self.data_path,audio_pickle_file], # (main, sub)  #audio_dir,
            'query': "*.wav",
            'load_fn': sf.read,
            'return_utt_id': return_utt_id,
            'subset_num': subset_num,
        }
        return SingleDataset(**params)
    
    
    # MODEL ARCHITECTURE
    def define_model(self):
        # generator
        generator = self._define_generator(self.model_type)
        self.model['generator'] = generator.to(self.device)
        # discriminator
        discriminator = self._define_discriminator(self.model_type)
        self.model['discriminator'] = discriminator.to(self.device)

        # optimizer
        self._define_optimizer_scheduler()
        #self._show_setting()
    

    def _define_generator(self, model_type):
        generator = generator_audiodec
       
        return generator(**self.config['generator_params'])
    

    def _define_discriminator(self, model_type):
        discriminator = discriminator_rir

       
        return discriminator()

 

    
    
    def _define_optimizer_scheduler(self):
        generator_optimizer_class = getattr(
            torch.optim, 
            self.config['generator_optimizer_type']
        )
        discriminator_optimizer_class = getattr(
            torch.optim, 
            self.config['discriminator_optimizer_type']
        )
     
        self.optimizer = {
            'generator': generator_optimizer_class(
                self.model['generator'].parameters(),
                **self.config['generator_optimizer_params'],
            ),
            'discriminator': discriminator_optimizer_class(
                self.model['discriminator'].parameters(),
                **self.config['discriminator_optimizer_params'],
            ),
           
        }

        generator_scheduler_class = getattr(
            torch.optim.lr_scheduler,
            self.config.get('generator_scheduler_type', "StepLR"),
        )
        discriminator_scheduler_class = getattr(
            torch.optim.lr_scheduler,
            self.config.get('discriminator_scheduler_type', "StepLR"),
        )
     
        self.scheduler = {
            'generator': generator_scheduler_class(
                optimizer=self.optimizer['generator'],
                **self.config['generator_scheduler_params'],
            ),
            'discriminator': discriminator_scheduler_class(
                optimizer=self.optimizer['discriminator'],
                **self.config['discriminator_scheduler_params'],
            ),
         
        }


    # CRITERIA
    def define_criterion(self):
        self.criterion = {
        'gen_adv': GeneratorAdversarialLoss(
            **self.config['generator_adv_loss_params']).to(self.device),
        'dis_adv': DiscriminatorAdversarialLoss(
            **self.config['discriminator_adv_loss_params']).to(self.device),
        }
        if self.config.get('use_feat_match_loss', False):
            self.criterion['feat_match'] = FeatureMatchLoss(
                **self.config.get('feat_match_loss_params', {}),
            ).to(self.device)

        if self.config.get('use_stft_loss', False) or self.config.get('use_stft_loss_rir', False):
            self.criterion['stft'] = MultiResolutionSTFTLoss(**self.config['stft_loss_params'],).to(self.device)
 
        if self.config.get('use_mse_loss', False) or self.config.get('use_mse_loss_rir', False):
            self.criterion['mse'] = TimeDomainMSELoss().to(self.device)
        if self.config.get('use_edc_loss', False) or self.config.get('use_edc_loss_rir', False):
            self.criterion['edc'] = EnergyDecayCurveLoss().to(self.device)

   


    # TRAINER
    def define_trainer(self):
        bands = [125, 250, 500, 1000, 2000, 4000]  # which frequency bands are we interested in
        filter_length = 16384  # a magic number, not need to tweak this much
        fs =16000
        # only generate filters once and keep using them, that means you need to know the samplerate beforehand or convert to a fixed samplerate
        filters = generate_complementary_filterbank(fc=bands, fs=fs, filter_order=4, filter_length=filter_length, power=True)
        filters = cp.asarray([[filters]])

        if self.train_mode in ['autoencoder']:
            trainer = TrainerAutoEncoder
        elif self.train_mode in ['vocoder']:
            trainer = TrainerVocoder
        # elif self.train_mode in ['denoise']:
        #     trainer = TrainerDenoise
        else:
            raise NotImplementedError(f"Train mode: {self.train_mode} is not supported for Trainer!")
        trainer_parameters = {}
        trainer_parameters['steps'] = 0
        trainer_parameters['epochs'] = 0
        trainer_parameters['filters'] = filters
        trainer_parameters['data_loader'] = self.data_loader
        trainer_parameters['model'] = self.model
        trainer_parameters['criterion'] = self.criterion
        trainer_parameters['optimizer'] = self.optimizer
        trainer_parameters['scheduler'] = self.scheduler
        trainer_parameters['config'] = self.config
        trainer_parameters['device'] = self.device
        self.trainer = trainer(**trainer_parameters)
    

    # MODEL INITIALIZATION
    def initialize_model(self):
        initial = self.config.get("initial", "") 
        if os.path.exists(self.resume): # resume from trained model
            self.trainer.load_checkpoint(self.resume)
            logging.info(f"Successfully resumed from {self.resume}.")
        elif os.path.exists(initial): # initial new model with the pre-trained model
            self.trainer.load_checkpoint(initial, load_only_params=True)
            logging.info(f"Successfully initialize parameters from {initial}.")
        else:
            logging.info("Train from scrach")
        # load the pre-trained encoder for vocoder training
        if self.train_mode in ['vocoder']:
            analyzer_checkpoint = self.config.get("analyzer", "")
            assert os.path.exists(analyzer_checkpoint), f"Analyzer {analyzer_checkpoint} does not exist!"
            analyzer_config = self._load_config(analyzer_checkpoint)
            self._initialize_analyzer(analyzer_config, analyzer_checkpoint)
    

    def _initialize_analyzer(self, config, checkpoint):
        model_type = config.get('model_type', 'symAudioDec')
        if model_type in ['symAudioDec', 'symAudioDecUniv']:
            analyzer = generator_audiodec
        else:
            raise NotImplementedError(f"Model type: {model_type} is not supported for the analyzer!")
        self.model['analyzer'] = analyzer(**config['generator_params']).to(self.device)
        self.model['analyzer'].load_state_dict(
            torch.load(checkpoint, map_location='cpu')['model']['generator'])
        logging.info(f"Successfully load analyzer from {checkpoint}.")


    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--exp_root", type=str, default="exp")
    parser.add_argument("--resume", default="", type=str, nargs="?",
        help='checkpoint file path to resume training. (default="")',
    )
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--disable_cudnn', choices=('True','False'), default='False', help='Disable CUDNN')
    args = parser.parse_args()
        
    # initial train_main
    train_main = TrainMain(args=args)   

    # get dataset
    train_main.initialize_data_loader()
    
    # define models, optimizers, and schedulers
    train_main.define_model()
    #################Have to modify below###############
    # define criteria
    train_main.define_criterion()

    # define trainer
    train_main.define_trainer()

    # model initialization
    train_main.initialize_model()

    # run training loop
    train_main.run()

if __name__ == "__main__":
    main()
