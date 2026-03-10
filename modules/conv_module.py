import torch
import torch.nn as nn  
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Tuple,Callable,Optional
import numpy as np
import matplotlib.pyplot as plt
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        
        return output.type_as(input)

class ConvFeatureExtraction(nn.Module):
    def __init__(self,conv_layers: List[Tuple[int, int, int]],dropout: float = 0.0,conv_bias: bool = False):
        super().__init__()
        ##assert mode in {"default", "layer_norm"}

        in_d = 1 ## input_channel size
        def block(n_in,n_out,k,stride,conv_bias=False,):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv
            
            return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
        
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl
            self.conv_layers.append(block(in_d,dim,k,
                                        stride,conv_bias=conv_bias))
            
            in_d = dim
    
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        B,C,N,p=x.shape
        x=x.view(B*C*N,1,p).contiguous()
        # BxT -> BxCxT
        ##x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        u=self.pool(x)
        u.contiguous()
        return u.view(B,C,N,-1)    

class depth_convolution(nn.Module):
    def __init__(self,input_features,out_features,kernel_size=3,groups=5):
        super().__init__() 
        self.n_in=input_features
        self.n_out=out_features
        self.k=kernel_size
        self.c=groups
        self.conv= nn.Conv1d(self.n_in,self.n_out,self.k,groups=self.c)
        
    def forward(self,x):
        B,C,N,p = x.shape
        x=x.view(-1,C,p)
        z=self.conv(x)
        
        return z.view(N,C,self.n_out//C,-1)
"""
test_x=torch.randn(1,3,3,512) ###b,c,n,p (1,512) as the temporal dimension
ts_conv_enc = depth_convolution(20,60,kernel_size=3,groups=1) 

ts_embed=ts_conv_enc(test_x)
print(ts_embed.shape)### per channel feature is hidden in the 60 features for 20 channel 3-features/channel

total_params = sum(p.numel() for p in ts_conv_enc.parameters())
print(f"Total number of parameters: {(total_params):.2f}")
"""
##Unit testig the conv_module
"""test_x=torch.randn(1,6,2,256)
##print(test_x.shape)
max_ch=test_x.shape[1]
max_N=test_x.shape[2]
patch_len=test_x.shape[3]
##conv_layers = [(128,7,1),(128,7,1),(256,5,2),(256,5,2),(512,3,2)]
actual_N=8
actual_ch=4
ts_token_mask=((torch.arange(max_N).unsqueeze(0))<actual_N).bool().to(torch.device(device))
ch_mask=((torch.arange(max_ch).unsqueeze(0))<actual_ch).bool().to(torch.device(device))
conv_layers = [(128,5,1),(64,3,1)]
conv_extractor=ConvFeatureExtraction(conv_layers,dropout=0.1)
convolved_x=conv_extractor(test_x)
print(f'convolved_x_shape:{convolved_x.shape}')

print(convolved_x[:,0,:,:5])
print(convolved_x.shape)

total_params = sum(p.numel() for p in conv_extractor.parameters())
print(f"Total number of parameters: {(total_params):.2f}")"""


