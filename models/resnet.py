from collections import OrderedDict

import torch.nn.functional as F
import torchvision
from torch import nn
import torch
import torch.nn as nn
import torch.fft
import torchvision as tv
import torchvision
from torchvision import datasets, models, transforms
from torch.nn.modules.utils import _pair
import math
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair


class SimAM(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)
    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale


class MLP(nn.Module):
    def __init__(self, num_features, expansion_factor, dropout):
        super().__init__()
        num_hidden = num_features * expansion_factor
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x


class TokenMixer(nn.Module):
    def __init__(self, num_features, image_size ,num_patches, expansion_factor, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.mlp = MLP(num_patches, expansion_factor, dropout)
        self.image_size = image_size        
        self.spatial_att = SpatialGate()

    def SpatialGate_forward(self, x):
        residual =x 
        BB, HH_WW, CC = x.shape
        HH =  WW = int(math.sqrt(HH_WW))
        x = x.reshape(BB, CC, HH, WW)
        x = self.spatial_att(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(BB, -1, CC)
        x = residual + x    
        return x

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        residual = x
        x_pre_norm = self.SpatialGate_forward((x))
        x = self.norm(x_pre_norm)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, num_features, num_patches)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, num_patches, num_features)
        out = x + residual
        return out


class ChannelMixer(nn.Module):
    def __init__(self, num_features, image_size, num_patches, expansion_factor, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.mlp = MLP(num_features, expansion_factor, dropout)
        self.image_size =image_size
        self.channel_att = ChannelGate(num_features, )

    def ChannelGate_forward(self, x):
        residual =x 
        BB, HH_WW, CC = x.shape
        HH =  WW = int(math.sqrt(HH_WW))       
        x = x.reshape(BB, CC, HH, WW)
        x = self.channel_att(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(BB, -1, CC)
        x = residual + x    
        return x

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        residual = x
        x_pre_norm = self.ChannelGate_forward(x)
        x = self.norm(x_pre_norm)
        x = self.mlp(x)
        # x.shape == (batch_size, num_patches, num_features)
        out = x + residual
        return out


class MixerLayer(nn.Module):
    def __init__(self, num_features, image_size, num_patches, expansion_factor, dropout):
        super().__init__()
        self.token_mixer = TokenMixer(
            num_patches, image_size, num_features, expansion_factor, dropout
        )
        self.channel_mixer = ChannelMixer(
            num_patches, image_size, num_features, expansion_factor, dropout
        )

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        # x.shape == (batch_size, num_patches, num_features)
        return x


def check_sizes(image_size, patch_size):
    sqrt_num_patches, remainder = divmod(image_size, patch_size)
    assert remainder == 0, "`image_size` must be divisibe by `patch_size`"
    num_patches = sqrt_num_patches ** 2
    return num_patches


class ARM_Mixer(nn.Module):
    def __init__(
        self,
        image_size=14,
        patch_size=1,
        in_channels=256,
        num_features=256,
        expansion_factor=3,        
        dropout=0.5,
    ):
        num_patches = check_sizes(image_size, patch_size)
        super().__init__()
        
        self.mixers = MixerLayer(num_patches, image_size, num_features, expansion_factor, dropout)
        self.simam = SimAM()

    def forward(self, x):
        residual = x
        BB, CC, HH, WW= x.shape
        patches = x.permute(0, 2, 3, 1)
        patches = patches.view(BB, -1, CC)
        # patches.shape == (batch_size, num_patches, num_features)
        embedding = self.mixers(patches)
        
        embedding_rearrange = embedding.reshape(BB, CC,HH,WW)
        embedding_final = embedding_rearrange + self.simam(x)+x
        return embedding_final
    

class Backbone(nn.Sequential):
    def __init__(self, resnet):
        super(Backbone, self).__init__()
        self.feature1 = nn.Sequential(resnet.conv1,
                                  resnet.bn1, resnet.relu,resnet.maxpool)
        self.layer1= nn.Sequential(resnet.layer1)
        self.layer2= nn.Sequential(resnet.layer2)
        self.layer3= nn.Sequential(resnet.layer3)
                                  
        self.out_channels = 1024
        
    def forward(self, x):
        feat = self.feature1(x)
        layer1=self.layer1(feat)
        layer2=self.layer2(layer1)
        layer3=self.layer3(layer2)

        return OrderedDict([["feat_res4", layer3]])


    
class Res5Head(nn.Sequential):
    def __init__(self, resnet):
        super(Res5Head, self).__init__()  
        self.layer4 = nn.Sequential(resnet.layer4)  # res5
        self.out_channels = [1024, 2048]
         
        self.mlP_model = ARM_Mixer(in_channels=256, image_size=14, patch_size=1)
       
        self.qconv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.qconv2 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1)
       
        
                
    def forward(self, x):
        qconv1 = self.qconv1(x)
        x_sc_mlp_feat=self.mlP_model(qconv1)
        qconv2 = self.qconv2(x_sc_mlp_feat)
        
        layer5_feat = self.layer4(qconv2)
 
    
        x_feat = F.adaptive_max_pool2d(qconv2, 1)

        feat = F.adaptive_max_pool2d(layer5_feat, 1)
        
        return OrderedDict([["feat_res4", x_feat], ["feat_res5", feat]])
    

def build_resnet(name="resnet50", pretrained=True):
    from torchvision.models import resnet
    resnet.model_urls["resnet50"] = "https://download.pytorch.org/models/resnet50-f46c3f97.pth"
    resnet_model = resnet.resnet50(pretrained=True)

    # freeze layers
    resnet_model.conv1.weight.requires_grad_(False)
    resnet_model.bn1.weight.requires_grad_(False)
    resnet_model.bn1.bias.requires_grad_(False)

    return Backbone(resnet_model), Res5Head(resnet_model)
