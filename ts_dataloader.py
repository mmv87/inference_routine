###Datapipeline for SFT.jsonl suitable for multi-variate and univariate timeseries
## Updated with the attention_mask (accounting for ts_token padding)
## dataloader to set the pipeline
### for the subset of the dataset
import os
###os.environ['HF_HOME']='D:/hf_cache'

from torch.utils.data import Dataset,DataLoader
import torch
import json
from transformers import AutoModelForCausalLM,AutoTokenizer
"""device ='cuda' if torch.cuda.is_available() else 'cpu'

abs_modelpath="D:/hf_cache/hub/models--microsoft--Phi-4-mini-reasoning/snapshots/0e3b1e2d02ee478a3743abe3f629e9c0cb722e0a"
##print('path_read')
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

model_name='./hub/microsoft/phi-4-mini-reasoning'
device ='cpu'"""

"""model=AutoModelForCausalLM.from_pretrained(abs_modelpath,local_files_only=True)
model.to(device)"""
###tokenizer=AutoTokenizer.from_pretrained(abs_modelpath,local_file_only=True)
"""input_text='The following timeseries in the model'
tokenized = tokenizer(input_text,return_tensors='pt',add_special_tokens=False)['input_ids'][0]"""
###add special_tokens to the tokenizer
"""special_token_dict={'pad_token':"<|pad|>","additional_special_tokens":['<ts>','<ts/>']}
tokenizer.add_special_tokens(special_token_dict)

sft_file='D:/Doctoral_research/code_implementation/synthetic_data.jsonl'"""

## Dataset class to get the pipeline for a sample
## requirements for Dataset 
    ##1. To patchify the timeseries data (from 1D 1, T)---> (N*C,T)
    ##2. padding to the ts_tokens not requires
    ##3.return the actual_N indices per channel
    ### if the max_N and max_ch is fixed the indices of te assembled ts_tokens are fixed
    
class ts_textual(Dataset): 
    def __init__(self,patch_len,stride,tokenizer,file,device=device):
        super().__init__()
        self.patch_len=patch_len
        self.stride=stride
        self.tokenizer=tokenizer
        self.file=file
        self.device =device
        self.dataset=[]
        
        with open(self.file,'r',encoding='utf-8') as f:
            for line in f:
                obj=json.loads(line)
                self.dataset.append(obj)
        ###self.sliced_offset=self.byte_offset[:10000]

    def __len__(self):
        return len(self.dataset)
    
    def pad_and_patchify(self,ts_input:list,p,s):
        seq_len_list=[]
        pad_pattern=torch.tensor([0.0,0.0],dtype=torch.float16)
        ###ts_type=None
        if len(ts_input)>1 : ##multivariable case
            
            ##check if the individual tensors are same shape
            for metric in ts_input:
                seq_len_list.append(torch.tensor(metric).shape[0]) ###get the list of tensors 
                
            if max(seq_len_list)!=min(seq_len_list):
                ##print('staggered')
                ##ts_type='staggered' 
                ts_padded_list=[]
                ###loop through the channels to pad
                for metric in ts_input:
                    ts_univariate_tensor=torch.tensor(metric).squeeze(-1).unsqueeze(0) ##reshape to (1,seq_len)
                    ##ts_univariate_tensor=ts_univariate_tensor.
                    pad_width =max(seq_len_list)-ts_univariate_tensor.shape[1]
                    repeats=pad_width//2
                    pad_repeat=pad_pattern.repeat(repeats)
                    ts_uni_padded=torch.cat([ts_univariate_tensor,pad_repeat.view(1,-1)],dim=1)
                    ts_padded_list.append(ts_uni_padded) ##list of tensors in a multivariate channel
                    
                ts_local_padded=torch.cat(ts_padded_list)
                ts_local_padded=ts_local_padded.unsqueeze(-1)
                seq_len=ts_local_padded.shape[1]
                ##apply second_level padding
                if (seq_len%p)==0:      ##zero_padding
                    pad_width=0
                    pad_repeat=pad_width//2
                
                elif seq_len<p:         ##pad_length > seq_len
                    ##pad to seq_len
                    pad_width=p-seq_len
                    pad_repeat=pad_width//2 
                    
                else:
                    ##padding case
                    pad_width=p-(seq_len%p)
                    pad_repeat=pad_width//2
                
                padding_pattern=pad_pattern.repeat(pad_repeat)
                padding_pattern=padding_pattern.view(1,-1,1)
                pattern=padding_pattern.repeat(ts_local_padded.size(0), 1, ts_local_padded.size(2))
                ts_l2_padded =torch.cat([ts_local_padded,pattern],dim=1)
        
                ts_patched=ts_l2_padded.unfold(dimension=1,size=p,step=s)
                ts_patched=ts_patched.view(ts_local_padded.shape[0],-1,p)
                
                ###logic to correct the stagger 
            else:
                ts_tensor=torch.tensor(ts_input)
                seq_len=ts_tensor.shape[1]
                if (seq_len%p)==0:      ##zero_padding
                    pad_width=0
                    pad_repeat=pad_width//2
                
                elif seq_len<p:         ##pad_length > seq_len
                    ##pad to seq_len
                    pad_width=p-seq_len
                    pad_repeat=pad_width//2 
                    
                else:
                    ##padding case
                    pad_width=p-(seq_len%p)
                    pad_repeat=pad_width//2
                    
                padding_pattern=pad_pattern.repeat(pad_repeat)
                padding_pattern=padding_pattern.view(1,-1,1)
                pattern=padding_pattern.repeat(ts_tensor.size(0), 1, ts_tensor.size(2))
                ts_padded =torch.cat([ts_tensor,pattern],dim=1)
                ##ts_padded=ts_padded.unsqueeze(-1)
                ts_patched=ts_padded.unfold(dimension=1,size=p,step=s)
                ts_patched=ts_patched.contiguous()
                ts_patched=ts_patched.view(ts_tensor.shape[0],-1,p)
            
                ##return ts_patched
        else:                ##univariate case
            
            ##print('univariate')
            ##ts_type='univariate'
            ts_tensor=torch.tensor(ts_input)
            ts_tensor=ts_tensor.squeeze(-1)
            ##print(ts_tensor.shape)
            seq_len=ts_tensor.shape[1]
            
            ##pad_width=(seq_len-p)%s
            if seq_len%p==0:
                pad_width=0
                pad_repeat=pad_width//2
            elif seq_len<p:
                pad_width=p-seq_len
                pad_repeat=pad_width//2 
            else:
                pad_width=p-seq_len%p
                pad_repeat=pad_width//2
                
            padding_pattern=pad_pattern.repeat(pad_repeat)
            padding_pattern=padding_pattern.view(1,-1)
            ##print(padding_pattern.shape)
            ##pattern=padding_pattern.repeat(ts_tensor.size(0), 1, ts_tensor.size(2))
            ts_padded =torch.cat([ts_tensor,padding_pattern],dim=1)
            ##print(ts_padded)
            ts_patched=ts_padded.unfold(1,p,s)
            ts_patched=ts_patched.contiguous()
            ##return ts_patched
            
        return ts_patched       
    
    def ts_pair_indices(self,tokenized):
        """tokenized= self.tokenizer(prompt,return_tensors='pt',add_special_tokens=False)
        input_ids= tokenized['input_ids'][0]"""
        ts_start_token=self.tokenizer.convert_tokens_to_ids('<ts>')
        ts_end_token=self.tokenizer.convert_tokens_to_ids('<ts/>')
        ts_position=[]
    
        ##data structure to save the <ts>,<ts/> tokens ,list of tuples
        for i,token_id in enumerate(tokenized.tolist()):
            if (token_id==ts_start_token):
                ts_position.append(('start',i))
            elif (token_id==ts_end_token):
                ts_position.append(('end',i))
                
        stack =[]
        ts_pairs=[]
        
        for j in range(len(ts_position)):
            pos,idx = ts_position[j]
            if pos=='start':
                stack.append(idx)
            elif stack and pos=='end':
                start=stack.pop(0)
                ts_pairs.append((start,idx))

        return ts_pairs,tokenized.shape[0] ##list of tuples
     
    def _calculate_ts_indices(self,ts_pairs,c_in,max_N,total_textual_tokens):
        ##to calculate the ts_indices and textual indices for a sample
        tensor_ts_pairs=(torch.tensor(ts_pairs))
        channel_indices=torch.arange(c_in,dtype=torch.long)
        ##offset_vec = (channel_indices*3)
        tensor_ts_pairs[:,:]+=channel_indices.view(-1,1)*max_N
        tensor_ts_pairs[:,1]+=max_N
        new_ts=(tensor_ts_pairs[:,0])
        offset_entries=(torch.arange(1,max_N+1).view(-1,1))
        ts_indices=(new_ts+offset_entries).t().flatten() ### indices for ts_patch insertions
        ###total_indices=torch.arange(1,40)
        T_new=total_textual_tokens+(c_in*max_N)
        is_ts_new=torch.zeros(T_new, dtype=torch.bool)
        is_ts_new[ts_indices]=True
        new_text_indices = torch.nonzero(~is_ts_new).squeeze()
        
        return ts_indices,new_text_indices,T_new
        
    def __getitem__(self,idx):
        """with open(self.file,'rb') as file:
            file.seek(self.byte_offset[idx])
            line =file.readline()
            sample =json.loads(line)
            """
        input = self.dataset[idx]['prompt']
        timeseries=self.dataset[idx]['timeseries'] ###list of lists
        
        input_ids=self.tokenizer(input,return_tensors='pt',add_special_tokens=False)['input_ids'][0]
        ###output_ids=self.tokenizer(output,return_tensors='pt',add_special_tokens=False)['input_ids'][0]
        ##print(f'test_output_ids:{output_ids}')
        ###combined_ids=torch.cat([input_ids,output_ids],dim=0)
        
        ###print(f'total_textual:{combined_ids.shape}')
        ts_patched = self.pad_and_patchify(timeseries,self.patch_len,self.stride)
        ch=ts_patched.shape[0]
        N=ts_patched.shape[1]
        ts_pairs,total_text_tokens=self.ts_pair_indices(input_ids)
        assert len(ts_pairs)==ch
        ts_tokens,text_tokens,total_tokens=self._calculate_ts_indices(ts_pairs,ch,N,total_text_tokens)
    
        ###assert labels.shape==combined_ids.shape
        ##attention_mask
        attention_mask=torch.ones(total_tokens,dtype=torch.long,device=self.device)
        ##attention_mask_batch.append(attention_mask)
        ##ts_pair_indices   
             
        return{"input_ids":input_ids,
            "ts_input":ts_patched,
            "attention_mask":attention_mask,
             "ts_indices":ts_tokens,
             "text_indices":text_tokens,
             "ts_pairs":torch.tensor(ts_pairs),
            }
###collate function
def collate_func(batch,tokenizer=None):
    input_ids = [x['input_ids'] for x in batch]
    ###labels_batch=[x['labels'] for x in batch]
    attention_mask_batch=[x['attention_mask'] for x in batch]
    padded_ts_data=[x['ts_input'] for x in batch] 
    ts_pairs=[x['ts_pairs'] for x in batch]
    ###assembler helper vars
    ts_indices =[x['ts_indices'] for x in batch] 
    text_indices=[x['text_indices'] for x in batch]
    
    return{
        'input_ids':torch.stack(input_ids),
        ###"labels":torch.stack(labels_batch),
        'attention_mask':torch.stack(attention_mask_batch),
        "time_series":torch.stack(padded_ts_data),
        "ts_indices":torch.stack(ts_indices),
        "textual_indices":torch.stack(text_indices),
        "ts_pairs":torch.stack(ts_pairs)}   ##list of tensor (bs,max_N,Patch_len)

###dataset=ts_textual(128,128,_json_path,tokenizer_modified,device=device,model_dtype=None)
##dataloader
"""dataset_for_test=ts_textual(128,128,tokenizer,sft_file,device=device)
dataloader=DataLoader(dataset_for_test,batch_size=5,shuffle=True,collate_fn=lambda b:collate_func(b,tokenizer=tokenizer))

##to check if Inference dataloader makes sense
for batch in dataloader:
  print(batch['time_series'].shape) ##shape -- [bs,c,N,P]
  print(batch['input_ids'].shape)
  print(batch['ts_pairs'].shape)
  print(batch['ts_indices'])
  print(batch['textual_indices'])
  ###print(batch['attention_mask'].shape)
  ###zprint(batch['labels'])
  break"""
