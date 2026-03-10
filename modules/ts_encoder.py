
###Main ts_encoder that fuses the conv_module and transformer 
###Concatenate the features from conv_module and ts_transformer module and project to the llm_backbone 
##from conv_module import ConvFeatureExtractionModel
##from ts_encoder_rel_bias import PatchTSTEncoder
##rom ts_encoder_rel_bias import PatchTSTEncoder
import torch.nn as nn  
import torch


class llm_projection(nn.Module):
    def __init__(self,conv_module,conv_features,trans_module,trans_embedding,d_fusion,d_llm):
        super().__init__()
        self.conv_module=conv_module
        self.trans_module=trans_module
        self.d_fusion=d_fusion
        self.d_llm=d_llm
        self.conv_features=conv_features
        self.trans_embedding=trans_embedding
        self.conv_proj=nn.Linear(self.conv_features,self.d_fusion)
        self.trans_proj=nn.Linear(self.trans_embedding,self.d_fusion)
        
        self.gate=nn.Linear(2*self.d_fusion,self.d_fusion)
        
        self.llm_projection=nn.Linear(self.d_fusion,self.d_llm)
        
    def forward(self,x):
        conv_embed= self.conv_module(x)
        trans_embed=self.trans_module(x)
        
        z_conv=self.conv_proj(conv_embed)
        z_trans=self.trans_proj(trans_embed)
        
        g=torch.sigmoid(self.gate(torch.cat([z_conv,z_trans],dim=-1)))
        z_gated=g*z_conv+(1.0-g)*z_trans
        z_llm = self.llm_projection(z_gated)
        
        return z_llm

"""
ts_text=torch.randn(1,3,2,256)
conv_layers = [(128,5,1),(64,3,1)]
patch_len=256
ts_transformer=PatchTSTEncoder(patch_len=patch_len,n_layers=2,d_model=512,n_heads=8,
                             shared_embedding=True,d_ff=1024,norm='Layer',attn_dropout=0.,dropout=0.1,activation='gelu',store_attn=False,res_attention=False,pre_norm=True,pe='zeros',learn_pe=True,verbose=False)
ts_conv_module=ConvFeatureExtractionModel(conv_layers,dropout=0.1)

ts_encoder = llm_projection(ts_conv_module,64,ts_transformer,512,1024,3072)

ts_embeddings=ts_encoder(ts_text)
print(ts_embeddings.shape)"""


