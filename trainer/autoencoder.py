#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)

"""Training flow of symmetric codec."""

import logging
import torch
from trainer.trainerGAN import TrainerVQGAN

import numpy as np
from wavefile import WaveWriter, Format
import os
import shutil
import torch.nn as nn
import torch.nn.functional as F
import cupy as cp
from cupyx.scipy.signal import fftconvolve


class Trainer(TrainerVQGAN):
    def __init__(
        self,
        steps,
        epochs,
        filters,
        data_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        config,
        device=torch.device("cpu"),
    ):
        super(Trainer, self).__init__(
           steps=steps,
           epochs=epochs,
           filters=filters,
           data_loader=data_loader,
           model=model,
           criterion=criterion,
           optimizer=optimizer,
           scheduler=scheduler,
           config=config,
           device=device,
        )
        self.fix_encoder = False
        self.paradigm = config.get('paradigm', 'efficient') 
        self.generator_start = config.get('start_steps', {}).get('generator', 0)
        self.discriminator_start = config.get('start_steps', {}).get('discriminator', 200000)
        self.filters = filters


    def _train_step(self, batch):
        """Single step of training."""
        mode = 'train'
        rs, rir = batch

        # print("disc_num ",disc_num)

        rs = rs.to(self.device)

        rir = rir.to(self.device)

        

        # check generator step
        if self.steps < self.generator_start:
            self.generator_train = False
        else:
            self.generator_train = True
            
        # check discriminator step
        if self.steps < self.discriminator_start:
            self.discriminator_train = False
        else:
            self.discriminator_train = True
    

        #######################
        #      Generator      #
        #######################
        if self.generator_train:
            # initialize generator loss
            gen_loss = 0.0

            # main genertor operation
            # y_rir_, zq_rir, z_rir, vqloss_rir, perplexity_rir,  = nn.parallel.data_parallel(self.model["generator"],(rs, color_image, speaker_view_location), self.gpus)
            y_rir_   = nn.parallel.data_parallel(self.model["generator"],(rs), self.gpus)
            
            # y_rir_cpu = cp.asarray(y_rir_.to("cpu").detach())

            

            # perplexity info
            # self._perplexity(perplexity_rir, label="rir", mode=mode)

            # vq loss
            # gen_loss += self._vq_loss(vqloss_rir, label="rir", mode=mode)
            
            # metric loss
            gen_loss += self._metric_loss_rir(y_rir_, rir, self.filters,mode=mode)
            

            # adversarial loss
            if self.discriminator_train:
               
                p_rir_ = nn.parallel.data_parallel(self.model["discriminator"],y_rir_,self.gpus)
                if self.config["use_feat_match_loss"]:
                    with torch.no_grad():
                        p_rir = nn.parallel.data_parallel(self.model["discriminator"],rir,self.gpus) 
                else:
                    p_rir = None
                gen_loss += self._adv_loss(p_rir_, p_rir, mode=mode)



            # update generator
            self._record_loss('generator_loss', gen_loss, mode=mode)
            self._update_generator(gen_loss)

        #######################
        #    Discriminator    #
        #######################
        if self.discriminator_train:
            # re-compute y_ which leads better quality
            with torch.no_grad():
                y_rir_ = nn.parallel.data_parallel(self.model["generator"],(rs), self.gpus) 

               
            
            p_rir = nn.parallel.data_parallel(self.model["discriminator"],rir,self.gpus) 
            p_rir_ = nn.parallel.data_parallel(self.model["discriminator"],y_rir_,self.gpus) 
            dis_loss_speech = self._dis_loss(p_rir_, p_rir, mode=mode)

            self._update_discriminator(dis_loss_speech)

        if(self.steps%5000==0):

            rir_path = os.path.join(self.save_path,"RIR")


            if(not os.path.exists(self.save_path)):
                os.mkdir(self.save_path)
                os.mkdir(rir_path)
            
            step_num = "step"+str(self.steps)

            rir_step_path = os.path.join(rir_path ,step_num)

        

            rir_step_path_real = os.path.join(rir_step_path ,"real_sample/")
            rir_step_path_fake = os.path.join(rir_step_path ,"fake_sample/")

            if(not os.path.exists(rir_step_path)):
                # shutil.rmtree(rir_step_path, ignore_errors=True)
                os.mkdir(rir_step_path)
                os.mkdir(rir_step_path_real)
                os.mkdir(rir_step_path_fake)




            for i in range(8): #(rir.shape[0]):
                
                real_RIR_path = rir_step_path_real +str(i)+".wav" 
                fake_RIR_path = rir_step_path_fake+str(i)+".wav"
                fs =16000
                real_IR = np.array(rir[i].to("cpu").detach())
                generated_IR = np.array(y_rir_[i].to("cpu").detach())

                r = WaveWriter(real_RIR_path, channels=1, samplerate=fs)
                r.write(np.array(real_IR))
                f = WaveWriter(fake_RIR_path, channels=1, samplerate=fs)
                f.write(np.array(generated_IR))

           

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()


    @torch.no_grad()
    def _eval_step(self, batch,steps):
        """Single step of evaluation."""
        mode = 'eval'
        rs, rir = batch

        rs = rs.to(self.device)

     
        rir = rir.to(self.device)

        
        # initialize generator loss
        gen_loss = 0.0

        # main genertor operation
        # y_rir_, zq_rir, z_rir, vqloss_rir, perplexity_rir = nn.parallel.data_parallel(self.model["generator"],(rs, color_image, speaker_view_location), self.gpus)

        y_rir_ = nn.parallel.data_parallel(self.model["generator"],(rs), self.gpus)

        
        

        # perplexity info
        # self._perplexity(perplexity_rir, label="rir", mode=mode)


        # vq_loss
        # gen_loss += self._vq_loss(vqloss_rir, label="rir", mode=mode)
        
        # metric loss
        gen_loss += self._metric_loss_rir(y_rir_, rir,self.filters, mode=mode)

        if self.discriminator_train:
            # adversarial loss           
            
            p_rir = nn.parallel.data_parallel(self.model["discriminator"],rir,self.gpus) 
            p_rir_ = nn.parallel.data_parallel(self.model["discriminator"],y_rir_,self.gpus) 
            gen_loss += self._adv_loss(p_rir_, p_rir, mode=mode)
            
             
            # discriminator loss

            self._dis_loss_speech(p_rir_, p_rir, mode=mode)
           

           

        # generator loss
        self._record_loss('generator_loss', gen_loss, mode=mode)


        
        rir_path = os.path.join(self.save_path,"RIR_Eval")


         
        step_num = "step"+str(steps)

        rir_step_path = os.path.join(rir_path ,step_num)

        

        rir_step_path_real = os.path.join(rir_step_path ,"real_sample/")
        rir_step_path_fake = os.path.join(rir_step_path ,"fake_sample/")


        if(not os.path.exists(rir_step_path)):
            # shutil.rmtree(rir_step_path, ignore_errors=True)
            os.mkdir(rir_step_path)
            os.mkdir(rir_step_path_real)
            os.mkdir(rir_step_path_fake)




        for i in range(8):#(rir.shape[0]):
            
            real_RIR_path = rir_step_path_real +str(i)+".wav" 
            fake_RIR_path = rir_step_path_fake+str(i)+".wav"
            fs =16000
            real_IR = np.array(rir[i].to("cpu").detach())
            generated_IR = np.array(y_rir_[i].to("cpu").detach())

            r = WaveWriter(real_RIR_path, channels=1, samplerate=fs)
            r.write(np.array(real_IR))
            f = WaveWriter(fake_RIR_path, channels=1, samplerate=fs)
            f.write(np.array(generated_IR))

     

        

       

