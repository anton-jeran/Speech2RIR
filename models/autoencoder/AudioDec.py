#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)

"""AudioDec model."""

import torch
import inspect

from layers.conv_layer import CausalConv1d, CausalConvTranspose1d
from models.autoencoder.modules.encoder import Encoder
from models.autoencoder.modules.decoder_rir import Decoder_RIR
# from models.autoencoder.modules.decoder_speech import Decoder_SPEECH
from models.vocoder.HiFiGAN import Generator as Decoder_SPEECH
from models.autoencoder.modules.projector import Projector
from models.autoencoder.modules.quantizer import Quantizer
from models.utils import check_mode


### GENERATOR ###
class Generator(torch.nn.Module):
    """AudioDec generator."""

    def __init__(
        self,
        input_channels=1,
        output_channels_rir=1,
        # output_channels_speech=1,
        # channels_sp=512,
        # kernel_size_sp=7,
        # upsample_scales_sp=(8, 8, 2, 2),
        # upsample_kernel_sizes_sp=(16, 16, 4, 4),
        # resblock_kernel_sizes_sp=(3, 7, 11),
        # resblock_dilations_sp=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        # groups_sp=1,
        # bias_sp=True,
        # use_additional_convs_sp=True,
        # nonlinear_activation_sp="LeakyReLU",
        # nonlinear_activation_params_sp={"negative_slope": 0.1},
        # use_weight_norm_sp=True,
        # stats_sp=None,
        encode_channels=16,
        decode_channels=16,
        code_dim=64,
        codebook_num=8,
        codebook_size=1024,
        bias=True,
        combine_enc_ratios=(2,  4),
        # seperate_enc_ratios_speech=(8, 16, 32),
        seperate_enc_ratios_rir=(8, 12, 16, 32),
        rir_dec_ratios=(128, 64, 32, 16, 8, 4),
        combine_enc_strides=(2, 2),
        # seperate_enc_strides_speech=(3, 5, 5),
        seperate_enc_strides_rir=(3, 5, 5, 5),
        rir_dec_strides=(5, 5, 5, 3, 2, 2),
        mode='causal',
        codec='audiodec',
        projector='conv1d',
        quantier='residual_vq',
    ):
        super().__init__()
        if codec == 'audiodec':
            encoder = Encoder
            # decoder_speech = Decoder_SPEECH
            decoder_rir = Decoder_RIR

        else:
            raise NotImplementedError(f"Codec ({codec}) is not supported!")
        self.mode = mode
        self.input_channels = input_channels

        self.encoder = encoder(
            input_channels=input_channels,
            encode_channels=encode_channels,
            combine_channel_ratios=combine_enc_ratios,
            # seperate_channel_ratios_speech=seperate_enc_ratios_speech,
            seperate_channel_ratios_rir=seperate_enc_ratios_rir,
            combine_strides=combine_enc_strides,
            # seperate_strides_speech=seperate_enc_strides_speech,
            seperate_strides_rir=seperate_enc_strides_rir,
            kernel_size=7,
            bias=bias,
            mode=self.mode,
        )


        # self.decoder_speech = decoder_speech(
        #     in_channels=code_dim,
        #     out_channels=output_channels_speech,
        #     channels=channels_sp,
        #     kernel_size=kernel_size_sp,
        #     upsample_scales=upsample_scales_sp,
        #     upsample_kernel_sizes=upsample_kernel_sizes_sp,
        #     resblock_kernel_sizes=resblock_kernel_sizes_sp,
        #     resblock_dilations=resblock_dilations_sp,
        #     groups=groups_sp,
        #     bias=bias_sp,
        #     use_additional_convs=use_additional_convs_sp,
        #     nonlinear_activation=nonlinear_activation_sp,
        #     nonlinear_activation_params=nonlinear_activation_params_sp,
        #     use_weight_norm=use_weight_norm_sp,
        #     stats=stats_sp,
        # )

        # self.projector_speech = Projector(
        #     input_channels=self.encoder.out_channels,
        #     code_dim=code_dim,
        #     kernel_size=3,
        #     stride=1,
        #     bias=False,
        #     mode=self.mode,
        #     model=projector,
        # )

        # self.quantizer_speech = Quantizer(
        #     code_dim=code_dim,
        #     codebook_num=codebook_num,
        #     codebook_size=codebook_size,
        #     model=quantier,
        # )

        self.decoder_rir = decoder_rir(
            code_dim=32* encode_channels  ,#code_dim,
            output_channels=output_channels_rir,
            decode_channels=decode_channels,
            channel_ratios=rir_dec_ratios,
            strides=rir_dec_strides,
            kernel_size=7,
            bias=bias,
            mode=self.mode,
        )

        self.projector_rir = Projector(
            input_channels=self.encoder.out_channels,
            code_dim=code_dim,
            kernel_size=3,
            stride=1,
            bias=False,
            mode=self.mode,
            model=projector,
        )

        self.quantizer_rir = Quantizer(
            code_dim=code_dim,
            codebook_num=codebook_num,
            codebook_size=codebook_size,
            model=quantier,
        )



    def forward(self, x):

        (batch, channel, length) = x.size()

       


        if channel != self.input_channels: 
            x = x.reshape(-1, self.input_channels, length) # (B, C, T) -> (B', C', T)
       
        x_rir = self.encoder(x)

        # print("x shape  ",x.shape)
        # print("x_rir shape  ",x_rir.shape)


        # print("z_speech shape  ",z_speech.shape)
        # print("zq_speech shape  ",zq_speech.shape)

        # z_rir = self.projector_rir(x_rir)
        # zq_rir, vqloss_rir, perplexity_rir = self.quantizer_rir(z_rir)

        # print("z_rir shape  ",z_rir.shape)
        # print("zq_rir shape  ",zq_rir.shape)
        # print("zq_rir shape 1 ",zq_rir.shape[1])


        y_rir = self.decoder_rir(x_rir)

        # print("y_rir shape  ",y_rir.shape)

        # input("summa 123")

        # return y_speech, y_rir, zq_speech, zq_rir, z_speech, z_rir, vqloss_speech, vqloss_rir, perplexity_speech, perplexity_rir
        return y_rir #, zq_rir, z_rir, vqloss_rir, perplexity_rir


