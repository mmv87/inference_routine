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
import numpy as np
from torch.nn.utils.rnn import pad_sequence
device ='cuda' if torch.cuda.is_available() else 'cpu'
"""
abs_modelpath="D:/hf_cache/hub/models--microsoft--Phi-4-mini-reasoning/snapshots/0e3b1e2d02ee478a3743abe3f629e9c0cb722e0a"
##print('path_read')
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
model_name='./hub/microsoft/phi-4-mini-reasoning'
device ='cpu'
print(device)"""
"""model=AutoModelForCausalLM.from_pretrained(abs_modelpath,local_files_only=True)
model.to(device)"""

"""tokenizer=AutoTokenizer.from_pretrained(abs_modelpath,local_file_only=True)
special_token_dict={'pad_token':"<|pad|>","additional_special_tokens":['<ts>','<ts/>']}
tokenizer.add_special_tokens(special_token_dict)"""

##align_256_file='D:/Doctoral_researh/code_implementation/Time_series_reasoning/training_dataset/ChatTS-Training-Dataset/align_256/train.jsonl'"""
##univar_eval="D:/Doctoral_research/code_implementation/Time_series_reasoning/dataset/uni_local.jsonl"
##multi_local_eval="D:/Doctoral_research/code_implementation/Time_series_reasoning/dataset/multi_local.jsonl"
###sft_file='D:/Doctoral_research/code_implementation/Time_series_reasoning/training_dataset/ChatTS-Training-Dataset/sft/sft_train.jsonl'"""
##print(align_256_file)
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
        self.byte_offset=[]
        
        if self.file.endswith(".json"):
            with open(self.file,'r',encoding='utf-8') as file:
                self.dataset=json.load(file)
        else:
            ###modify to .jsonl format
            with open(self.file,'r',encoding='utf-8') as f:
                for line in f:
                    self.dataset.append(json.loads(line))
        
    ##self.sliced_offset=self.byte_offset[:5000]

    def __len__(self):
        return len(self.dataset)
    
    def sp_encoding(self,timeseries):
        ##logic to get the normalize and get the 
        meta_prompts=[]
        timeseries_list=[]
        for ts_data in timeseries:
            mean = np.mean(ts_data)
            scaled_timeseries = ts_data - mean
            scale_factor = 1.0
            if np.any(np.abs(scaled_timeseries) >= 3.0):
                scale_factor = np.max(np.abs(scaled_timeseries)) / 3.0
                scaled_timeseries /= scale_factor
            # meta-prompt
            meta_prompt = f"[Value Offset: {mean:.4f}|Value Scaling: {scale_factor:.4f}]"
            meta_prompt_tokens=self.tokenizer(meta_prompt,return_tensors='pt')['input_ids']
            ##print(f'meta_shape{meta_prompt_tokens.shape}')
            meta_prompts.append(meta_prompt_tokens)
            list_ts=scaled_timeseries.tolist()
            ###print(f'list_ts:{len(list_ts)}')
            timeseries_list.append(list_ts)
            
        ###print(f'meta_prompts:{len(meta_prompts)}')
        return timeseries_list,meta_prompts
    
    def insert_meta_prompt(self,sequence:torch.Tensor,meta_prompts:list,ts_start):
        current_offset = 0
        result = sequence.clone() if torch.is_tensor(sequence) else list(sequence)
        result.unsqueeze_(0)
        ###print(f'result_seq:{result.shape}')
        ts_start_indices=ts_start.tolist()
        
        for i, original_pos in enumerate(ts_start_indices):
            ##print(f'orginal_ts_start:{original_pos}')
            actual_pos = original_pos + current_offset
            # Get the specific meta_prompt for this channel/pair
            meta = meta_prompts[i]
            meta_len = meta.shape[1]
            # Perform Splice
            if torch.is_tensor(result):
                result = torch.cat([result[:,:actual_pos],meta,result[:,actual_pos:]],dim=1)
            else:
                result = result[:,:actual_pos] + meta + result[:actual_pos:]
            # Update the offset for the NEXT iteration
            current_offset += meta_len
        
        ##print(f'total_textual_len:{result.shape[1]}')
        return result,result.shape[1]
    
    def pad_and_patchify(self,ts_input:list,p,s):
        seq_len_list=[]
        pad_pattern=torch.tensor([0.0,0.0],dtype=torch.float16)
        ###ts_type=None
        if len(ts_input)>1 : ##multivariable case
            ##check if the individual tensors are same shape
            for metric in ts_input:
                seq_len_list.append(torch.tensor(metric).shape[0]) ###get the list of tensors 
                ##print(torch.tensor(metric).shape)
            if max(seq_len_list)!=min(seq_len_list):
                ##print('staggered')
                print(max(seq_len_list),min(seq_len_list))
                ##ts_type='staggered' 
                ts_padded_list=[]
                ###remove the stagger
                for metric in ts_input:
                    ts_padded_list.append(torch.tensor(metric))
                ts_uniform=pad_sequence(ts_padded_list,batch_first=True,padding_value=0)
                ch_dim=ts_uniform.shape[0]
                seq_len=ts_uniform.shape[1]
                assert ts_uniform.shape[1]==max(seq_len_list)
                """##ts_univariate_tensor=torch.tensor(metric).squeeze(-1).unsqueeze(0) ##reshape to (1,seq_len)
                    ##ts_univariate_tensor=ts_univariate_tensor.
                    pad_width =max(seq_len_list)-ts_univariate_tensor.shape[1]
                    repeats=torch.zeros()
                    repeats=pad_width//2
                    pad_repeat=pad_pattern.repeat(repeats)
                    ts_uni_padded=torch.cat([ts_univariate_tensor,pad_repeat.view(1,-1)],dim=1)
                    ts_padded_list.append(ts_uni_padded) ##list of tensors in a multivariate channel"""
                """ts_local_padded=torch.cat(ts_padded_list)
                ts_local_padded=ts_local_padded.unsqueeze(-1)
                seq_len=ts_local_padded.shape[1]"""
                ##apply second_level padding
                if (seq_len%p)==0:      ##zero_padding
                    pad_width=0
                    pad_repeat=pad_width
                elif seq_len<p:         ##pad_length > seq_len
                    ##pad to seq_len
                    pad_width=p-seq_len
                    pad_repeat=pad_width
                else:
                    ##padding case
                    pad_width=p-(seq_len%p)
                    pad_repeat=pad_width
                    
                if (pad_repeat!=0):
                    padding_pattern=torch.zeros((ch_dim,pad_repeat))
                    ts_padded =torch.cat([ts_uniform,padding_pattern],dim=1)
                else:
                    ts_padded=ts_uniform.clone()
                
                ts_patched=ts_padded.unfold(dimension=1,size=p,step=s)
                ts_patched=ts_patched.contiguous()
                ts_patched=ts_patched.view(ts_uniform.shape[0],-1,p)
                """
                padding_pattern=pad_pattern.repeat(pad_repeat)
                padding_pattern=padding_pattern.view(1,-1,1)
                pattern=padding_pattern.repeat(ts_local_padded.size(0), 1, ts_local_padded.size(2))
                ts_l2_padded =torch.cat([ts_local_padded,pattern],dim=1)
        
                ts_patched=ts_l2_padded.unfold(dimension=1,size=p,step=s)
                ts_patched=ts_patched.view(ts_local_padded.shape[0],-1,p)"""
                ###logic to correct the stagger 
            else:
                print('uniform')
                ts_tensor=torch.tensor(ts_input)
                ##print(ts_tensor.shape)
                ###ts_tensor.unsqueeze_(-1)
                ##print(f'ts_tensor_shape:{ts_tensor.shape}')
                seq_len=ts_tensor.shape[1]
                ch_dim=ts_tensor.shape[0]
                ###print(f'seq_len:{seq_len}')
                if (seq_len%p)==0:      ##zero_padding
                    pad_width=0
                    pad_repeat=pad_width
                
                elif (seq_len<p):         ##patch_len > seq_len
                    ##pad to seq_len
                    pad_width=p-seq_len
                    pad_repeat=pad_width 
                    
                else:
                    ##padding case
                    pad_width=p-(seq_len%p)
                    pad_repeat=pad_width

                ##print(f'pad_repeat{pad_repeat}')
                if (pad_repeat!=0):
                    padding_pattern=torch.zeros((ch_dim,pad_repeat))
                    ts_padded =torch.cat([ts_tensor,padding_pattern],dim=1)
                else:
                    ts_padded=ts_tensor.clone()
                    
                ts_patched=ts_padded.unfold(dimension=1,size=p,step=s)
                ts_patched=ts_patched.contiguous()
                ts_patched=ts_patched.view(ts_tensor.shape[0],-1,p)
                ##return ts_patched
        else:                ##univariate case
            ##print('univariate')
            ts_tensor=torch.tensor(ts_input)
            ##print(ts_tensor.shape)
            seq_len=ts_tensor.shape[1]
            ch_dim=ts_tensor.shape[0] ###ch=1 i.e univariate
            ##pad_width=(seq_len-p)%s
            
            if (seq_len%p==0):
                pad_width=0
                pad_repeat=pad_width
            elif seq_len<p:
                pad_width=p-seq_len
                pad_repeat=pad_width
            else:
                pad_width=p-seq_len%p
                pad_repeat=pad_width
            
            if (pad_repeat!=0):
                padding_pattern=torch.zeros((ch_dim,pad_repeat))
                ts_padded =torch.cat([ts_tensor,padding_pattern],dim=1)
            else:
                ts_padded=ts_tensor.clone()
            
            ts_patched=ts_padded.unfold(1,p,s)
            ts_patched=ts_patched.contiguous()
            ts_patched=ts_patched.view(ts_tensor.shape[0],-1,p)
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
        tot_multimodal_tokens=total_textual_tokens+(c_in*max_N)
        is_ts_new=torch.zeros(tot_multimodal_tokens, dtype=torch.bool)
        is_ts_new[ts_indices]=True
        new_text_indices = torch.nonzero(~is_ts_new).squeeze()
        
        return ts_indices,new_text_indices,tot_multimodal_tokens
        
    def __getitem__(self,idx):
        ##self.byte_offset[idx]
        """with open(self.file,'rb') as file:
            file.seek(self.sliced_offset[idx])
            line =file.readline()
            sample =json.loads(line)"""
        input = self.dataset[idx]['question']
        #print(input)
        ###output = self.dataset[idx]['output']
        timeseries=self.dataset[idx]['timeseries'] ###list of lists
        prompt=f"""<|system|>You are a timeseries expect,answer briefly to the following question based on the timeseries data provided,
                    <|end|><|user|>{input}<|end|><|assistant|>"""
                
        ##print(prompt)  
        """messages = [
            {"role": "system", "content": "You are time series expert and answer the following question based on the timeseries provided"},
            {"role": "user", "content":f'{input}'}
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True,
            return_tensors="pt"
        ).squeeze(0)"""
        
        input_ids=self.tokenizer(prompt,return_tensors='pt',add_special_tokens=False)['input_ids'][0]
        #print(f'input_ids:{input_ids}')
        ##normalize the ts_data
        norm_ts,meta_prompt = self.sp_encoding(timeseries)
        ts_pairs,text_tokens_pre_meta_prompt=self.ts_pair_indices(input_ids)
        print(ts_pairs)
        ts_start=torch.tensor(ts_pairs)[:,0]        
        new_text_tokens,total_text_tokens=self.insert_meta_prompt(input_ids,meta_prompt,ts_start)
        ###print(f'total_textual:{new_text_tokens.shape}')
        ts_patched =self.pad_and_patchify(norm_ts,self.patch_len,self.stride)
        ch=ts_patched.shape[0]
        N=ts_patched.shape[1]
        assert len(ts_pairs)==ch
        ts_tokens,text_tokens,total_multlimodal_tokens=self._calculate_ts_indices(ts_pairs,ch,N,total_text_tokens)
        ##labels
        ##attention_mask
        attention_mask=torch.ones(total_multlimodal_tokens,dtype=torch.long,device=self.device)
        ##attention_mask_batch.append(attention_mask)             
        return{"input_ids":new_text_tokens,
            ##"output_ids":output_ids,
            "ts_input":ts_patched,
            ##"labels":labels,
            "attention_mask":attention_mask,
             "ts_indices":ts_tokens,
             "text_indices":text_tokens,
             "ts_pairs":torch.tensor(ts_pairs)}

###collate function
def collate_func(batch,tokenizer=None):
    input_ids = [x['input_ids'] for x in batch]
    ##labels_batch=[x['labels'] for x in batch]
    attention_mask_batch=[x['attention_mask'] for x in batch]
    padded_ts_data=[x['ts_input'] for x in batch] 
    ts_pairs=[x['ts_pairs'] for x in batch]
    ###assembler helper vars
    ts_indices =[x['ts_indices'] for x in batch] 
    text_indices=[x['text_indices'] for x in batch]
    
    return{
        'input_ids':torch.cat(input_ids),
        ##"labels":torch.stack(labels_batch),
        'attention_mask':torch.stack(attention_mask_batch),
        "time_series":torch.stack(padded_ts_data),
        "ts_indices":torch.stack(ts_indices),
        "textual_indices":torch.stack(text_indices),
        "ts_pairs":torch.stack(ts_pairs)}   ##list of tensor (bs,max_N,Patch_len)

"""
dataset_for_test=ts_textual(128,128,tokenizer,multi_local_eval,device=device)
dataloader=DataLoader(dataset_for_test,batch_size=1,shuffle=False,collate_fn=lambda b:collate_func(b,tokenizer=tokenizer))
###input_embeds = model.get_input_embeddings()
for idx,batch in enumerate(dataloader):
    if idx<5:
        print(batch['time_series'].shape)
        print(tokenizer.decode(batch['input_ids'][0]))
        print(f"input_ids:{batch['input_ids'].shape}")
        print(f"attention:{batch['attention_mask'].shape}")
    else:break"""
    
