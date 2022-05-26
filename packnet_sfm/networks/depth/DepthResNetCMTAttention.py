# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch.nn as nn
from functools import partial

from packnet_sfm.networks.layers.resnet.resnet_encoder_h import ResnetEncoder
from packnet_sfm.networks.depth.cmt_origin_cascade_h import CMT, CMT_Ti,CMT_XS, CMT_XS2, CMT_B, CMT_Layer
from packnet_sfm.networks.layers.resnet.depth_decoder_attention import DepthDecoder
from packnet_sfm.networks.layers.resnet.layers import disp_to_depth

########################################################################################################################

class fcconv(nn.Module):
    def __init__(self,  in_channel, out_channel):
        super().__init__()
        self.conv =  nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.nonlin = nn.ELU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(self.bn1(out))
        ##norm?
        return out

class DepthResNetCMTAttention(nn.Module):
    """
    Inverse depth network based on the ResNet architecture.

    Parameters
    ----------
    version : str
        Has a XY format, where:
        X is the number of residual layers [18, 34, 50] and
        Y is an optional ImageNet pretrained flag added by the "pt" suffix
        Example: "18pt" initializes a pretrained ResNet18, and "34" initializes a ResNet34 from scratch
    kwargs : dict
        Extra parameters
    """
    def __init__(self, version=None, **kwargs):
        super().__init__()
        assert version is not None, "DispResNet needs a version"

        num_layers = int(version[:2])       # First two characters are the number of layers
        pretrained = version[2:] == 'pt'    # If the last characters are "pt", use ImageNet pretraining
        assert num_layers in [18, 34, 50], 'ResNet version {} not available'.format(num_layers)

        self.resnet_encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained)

        self.stem_channel = 64
        self.embed_dim= 46    
        self.in_channels=[self.embed_dim, self.embed_dim *2 , self.embed_dim*4, self.embed_dim * 8]      
        self.de_channels=[64, 64, 128, 256, 512]
        
        #self.cmt = CMT_Ti(in_channels = 3, input_size = 256, embed_dim= self.embed_dim, stem_channels= self.stem_channel)
        
        input_width = 640
        input_height = 192
        repeats= [2,2,2,2]
        
        self.cmt_start_layer = 3
        self.use_upconv = False
        self.use_cmt_feature = False
        
        self.cmt = CMT_Layer(input_width = input_width, input_height= input_height, embed_dim= self.embed_dim, start_layer=self.cmt_start_layer, use_upconv =self.use_upconv or self.use_cmt_feature, repeats=repeats)


        for i in range(self.cmt_start_layer,5):
            value = 2**(i-1)
            self.de_channels[i] = self.embed_dim * value
            
        # self.stem_channel = 64
        # self.embed_dim= 52    
        # self.in_channels=[self.embed_dim, self.embed_dim *2 , self.embed_dim*4, self.embed_dim * 8]      
        # self.de_channels=[64, self.resnet_encoder.num_ch_enc[1] , self.embed_dim *2 , self.embed_dim*4, self.embed_dim * 8]      
        # self.cmt = CMT_XS(in_channels = 3, input_size = 256, embed_dim= self.embed_dim)

        # self.stem_channel = 64
        # self.embed_dim= 52    
        # self.in_channels=[self.embed_dim, self.embed_dim *2 , self.embed_dim*4, self.embed_dim * 8]      
        # self.de_channels=[64, self.resnet_encoder.num_ch_enc[1] , self.embed_dim *2 , self.embed_dim*4, self.embed_dim * 8]      
        # self.cmt = CMT_XS2(in_channels = 3, input_size = 256, embed_dim= self.embed_dim)

        # self.stem_channel = 64
        # self.embed_dim= 76    
        # self.in_channels=[self.embed_dim, self.embed_dim *2 , self.embed_dim*4, self.embed_dim * 8]      
        # self.de_channels=[64, self.resnet_encoder.num_ch_enc[1] , self.embed_dim *2 , self.embed_dim*4, self.embed_dim * 8]      
        # self.cmt = CMT_B(in_channels = 3, input_size = 256, embed_dim= self.embed_dim, stem_channels= self.stem_channel)


        self.upconv = fcconv(self.resnet_encoder.num_ch_enc[1],self.embed_dim)
        self.decoder = DepthDecoder(num_ch_enc=self.de_channels)
        self.scale_inv_depth = partial(disp_to_depth, min_depth=0.1, max_depth=100.0)

    def forward(self, rgb):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        features = self.resnet_encoder(rgb)
        #out = self.upconv(resnet_out[-1])
        if self.use_upconv:
            out = self.upconv(features[-1])
        else:
            out = features[-1]
        
        if self.cmt_start_layer>2:
            post_matching_feats = self.resnet_encoder.layer2(out)                     
            features.append(post_matching_feats)
            if self.use_upconv:
                out = self.upconv2(post_matching_feats)
            else:
                out = post_matching_feats       
                
            if self.cmt_start_layer>3:             
                post_matching_feats = self.resnet_encoder.layer3(post_matching_feats)
                features.append(post_matching_feats)
                if self.use_upconv:
                    out = self.upconv3(post_matching_feats)
                else:
                    out = post_matching_feats     
                    
                    
        out = self.cmt(out)    
        x = features + out
        x = self.decoder(x)
        disps = [x[('disp', i)] for i in range(4)]

        if self.training:
            return {
                'inv_depths': [self.scale_inv_depth(d)[0] for d in disps],
            }
        else:
            return {
                'inv_depths': self.scale_inv_depth(disps[0])[0],
            }

########################################################################################################################
