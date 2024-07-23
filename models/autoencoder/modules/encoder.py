#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://ieeexplore.ieee.org/document/9625818)

"""Encoder modules."""

import torch
import torch.nn as nn
import torchvision
import inspect

from layers.conv_layer import NonCausalConv1d
from layers.conv_layer import CausalConv1d
from models.autoencoder.modules.residual_unit import NonCausalResidualUnit
from models.autoencoder.modules.residual_unit import CausalResidualUnit
from models.utils import check_mode
from models.autoencoder.modules.horizonet import HorizonNet





class VISURAL_RES_NET(torch.nn.Module):
    def __init__(self):
        super(VISURAL_RES_NET, self).__init__()
        
        conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        layers = list(torchvision.models.resnet18(pretrained=True).children())[1:-1]
        self.feature_extraction = torch.nn.Sequential(conv1, *layers)  # features before conv1x1
        # self.predictor = torch.nn.Sequential(torch.nn.Linear(512, 1024))

    def forward(self, inputs):
        audio_feature = self.feature_extraction(inputs).squeeze(-1).squeeze(-1)
        pred = audio_feature # self.predictor(audio_feature) #
        shape = pred.shape 

        pred = pred.reshape(shape[0],512,int(shape[1]/512))
        pred = torch.cat((pred,pred,pred,pred),2)
        # pred = torch.cat((pred,pred),1)
    

        return pred


def load_trained_model(Net, path):
        state_dict = torch.load(path, map_location='cpu')
        net = Net(**state_dict['kwargs'])
        net.load_state_dict(state_dict['state_dict'])
        return net


# class VISURAL_HORIZON_NET(torch.nn.Module):
#     def __init__(self):
#         super(VISURAL_HORIZON_NET, self).__init__()
        
#         args_pth = "models/horizonet/resnet50_rnn__zind.pth"

#         layers = list(load_trained_model(HorizonNet, args_pth).children())[0:-1] 
#         self.feature_extraction = torch.nn.Sequential(*layers)  
#         self.predictor = torch.nn.Sequential(torch.nn.Linear(512, 1))

#     def forward(self, inputs):
#         audio_feature1 = self.feature_extraction(inputs)
#         print("audio_feature1 shape", audio_feature1.shape)
#         audio_feature = self.feature_extraction(inputs).squeeze(-1).squeeze(-1)
#         pred = self.predictor(audio_feature)

#         return pred

# class Location_Encoder(torch.nn.Module):
#     def __init__(self):
#         super(Location_Encoder,self).__init__()
#         self.conv = nn.Conv1d(1,64,6,1,bias=False)

#     def forward(self,x):
#         encoded_loc = self.conv(x)
#         shape = encoded_loc.shape

#         encoded_loc = encoded_loc.reshape(shape[0],64,int(shape[1]/64))
#         encoded_loc = torch.cat((encoded_loc,encoded_loc,encoded_loc,encoded_loc,encoded_loc,encoded_loc,encoded_loc,encoded_loc),1)
#         encoded_loc = torch.cat((encoded_loc,encoded_loc),2)

#         return encoded_loc

#     def inference(self,x):
#         encoded_loc = self.conv.inference(x)
#         shape = encoded_loc.shape

#         encoded_loc = encoded_loc.reshape(shape[0],64,int(shape[1]/64))
#         encoded_loc = torch.cat((encoded_loc,encoded_loc,encoded_loc,encoded_loc,encoded_loc,encoded_loc,encoded_loc,encoded_loc),1)
#         encoded_loc = torch.cat((encoded_loc,encoded_loc),2)

#         return encoded_loc


class EncoderBlock(torch.nn.Module):
    """ Encoder block (downsampling) """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        dilations=(1, 3, 9),
        bias=True,
        mode='causal',
    ):
        super().__init__()
        self.mode = mode
        if self.mode == 'noncausal':
            ResidualUnit = NonCausalResidualUnit
            Conv1d = NonCausalConv1d
        elif self.mode == 'causal':
            ResidualUnit = CausalResidualUnit
            Conv1d = CausalConv1d
        else:
            raise NotImplementedError(f"Mode ({self.mode}) is not supported!")

        self.res_units = torch.nn.ModuleList()
        for dilation in dilations:
            self.res_units += [
                ResidualUnit(in_channels, in_channels, dilation=dilation)]
        self.num_res = len(self.res_units)

        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(2 * stride),
            stride=stride,
            bias=bias,
        )
        
    def forward(self, x):
        for idx in range(self.num_res):
            x = self.res_units[idx](x)
        x = self.conv(x)
        return x
    
    def inference(self, x):
        check_mode(self.mode, inspect.stack()[0][3])
        for idx in range(self.num_res):
            x = self.res_units[idx].inference(x)
        x = self.conv.inference(x)
        return x


class Encoder(torch.nn.Module):
    def __init__(self,
        input_channels,
        encode_channels,
        combine_channel_ratios=(2, 4),
        seperate_channel_ratios_speech=(8, 16,32),
        seperate_channel_ratios_rir=(8, 8, 16,32),
        combine_strides=(2, 2),
        seperate_strides_speech=(3, 5, 5),
        seperate_strides_rir=(2, 3, 5, 5, 5),
        kernel_size=7,
        bias=True,
        mode='causal',
    ):
        super().__init__()
        assert len(combine_channel_ratios) == len(combine_strides)
        assert len(seperate_channel_ratios_speech) == len(seperate_strides_speech)
        assert len(seperate_channel_ratios_rir) == len(seperate_strides_rir)
        self.mode = mode
        if self.mode == 'noncausal':
            Conv1d = NonCausalConv1d
        elif self.mode == 'causal':
            Conv1d = CausalConv1d
        else:
            raise NotImplementedError(f"Mode ({self.mode}) is not supported!")

        self.conv = Conv1d(
            in_channels=input_channels, 
            out_channels=encode_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            bias=False)

        self.conv_combine = Conv1d(
            in_channels=input_channels, 
            out_channels=input_channels, 
            kernel_size=3, 
            stride=1, 
            bias=False)

        self.combine_conv_blocks = torch.nn.ModuleList()
        self.seperate_conv_blocks_1 = torch.nn.ModuleList()
        # self.seperate_conv_blocks_2 = torch.nn.ModuleList()
        


        in_channels = encode_channels

        self.encode_RIR = torch.nn.Sequential(
            torch.nn.Conv1d(input_channels, 24*in_channels, 20001, 500, 10000, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 1024
            # torch.nn.Conv1d(16*in_channels, 24*in_channels, 41, 2, 20, bias=False),
            # torch.nn.BatchNorm1d(24*in_channels),
            # torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv1d(24*in_channels, 32*in_channels, 41, 2, 20, bias=False),
            torch.nn.BatchNorm1d(32*in_channels),
            torch.nn.LeakyReLU(0.2, inplace=True),
            
        )

        self.encode_RIR_s = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 16*in_channels, 9600, 459, 4800, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 1024
            torch.nn.Conv1d(16*in_channels, 32*in_channels, 41, 2, 20, bias=False),
            torch.nn.BatchNorm1d(32*in_channels),
            torch.nn.LeakyReLU(0.2, inplace=True),
            
        )

        for idx, stride in enumerate(combine_strides):
            out_channels = encode_channels * combine_channel_ratios[idx]
            self.combine_conv_blocks += [
                EncoderBlock(in_channels, out_channels, stride, bias=bias, mode=self.mode)]
            in_channels = out_channels

        seperate_in_channels =  in_channels

        in_channels=seperate_in_channels

        for idx, stride in enumerate(seperate_strides_speech):
            out_channels = encode_channels * seperate_channel_ratios_speech[idx]
            self.seperate_conv_blocks_1 += [
                EncoderBlock(in_channels, out_channels, stride, bias=bias, mode=self.mode)]
            in_channels = out_channels

        in_channels=seperate_in_channels

        # for idx, stride in enumerate(seperate_strides_rir):
        #     out_channels = encode_channels * seperate_channel_ratios_rir[idx]
        #     self.seperate_conv_blocks_2 += [
        #         EncoderBlock(in_channels, out_channels, stride, bias=bias, mode=self.mode)]
        #     in_channels = out_channels

        self.combine_num_blocks = len(self.combine_conv_blocks)
        self.seperate_num_blocks_speech = len(self.seperate_conv_blocks_1)
        # self.seperate_num_blocks_rir = len(self.seperate_conv_blocks_2)
        self.out_channels = out_channels

        self.visual_res_net = VISURAL_RES_NET()
        # self.location_encode = Location_Encoder()

        # self.visual_horizon_net = VISURAL_HORIZON_NET()
    
    def forward(self, x):

        # ci_code = self.visual_res_net(ci)
        # locations_code = self.location_encode(locations)
        # ci_loc = torch.cat((ci_code,locations_code),2)
        # ci_loc = torch.cat((ci_loc,ci_loc),1)


        
        # x_rir = self.conv_combine(x)
        x_rir = self.encode_RIR(x)


        # x_rir_AV =torch.cat((x_rir,ci_code),2)


        return x_rir #x_rir_AV
    
    def encode(self, x):
        check_mode(self.mode, inspect.stack()[0][3])
        

        # ci_code = self.visual_res_net(ci)
        # locations_code = self.location_encode(locations)
        # ci_loc = torch.cat((ci_code,locations_code),2)
        # ci_loc = torch.cat((ci_loc,ci_loc),1)

        
        # x_rir = self.conv_combine(x)
        x_rir = self.encode_RIR(x_rir)

        # print("x_rir shape : ",x_rir.shape)
        # print("locations_code shape : ",locations_code.shape)
        # print("ci_code shape : ",ci_code.shape)

        x_rir_AV =x_rir #torch.cat((x_rir,ci_loc),2)


        return x_rir_AV
    

