##to load the ts_encoder and trained LLM model for inference
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import os
import json
## input as the 'timeseries' and 'input' prompt
## dataset requirement 
## to retrieve the input_prompt + ts_input (after patching)<ts?,<ts/> positions that is required for assembly

##tokenizerloading
path=os.getcwd()
file_path="/home/mmk/projects/def-zonata/mmk/ts_stage2"
print(f'current_path:{path}')
tokenizer_path =os.path.join(file_path,'llm_tokenizer')
tokenizer_modified = AutoTokenizer.from_pretrained(tokenizer_path)

##self.ts_encoder=ts_encoder_mlp(self.max_patches,self.max_channel,self.P,self.embed_size,device=self.device)

device ='cuda' if torch.cuda.is_available() else 'cpu'


###step-1 : synthetic data generation
### step-2 : preprocess the data to set the pipeline for the ts_encoder
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import re
import torch
pattern ="<ts><ts/>"
def get_sine_curve(num_points):
    t=np.linspace(0,1,num_points)
    x=np.sin(2*np.pi*t)
  ##ts=x.reshape(-1,1)
    return x

def synthetic_ts(length,trend_slope,period,amp,noise):
    t=np.arange(length)
    trend=trend_slope*t
    seasonality=amp*np.cos(2*np.pi*t/period)
    noise=noise*np.random.randn(length)
    ts=trend+seasonality+noise
    return ts
    
def inject_point_anomalies(ts:np.array, n_anomalies=3, magnitude=15.0):
    ts = ts.copy()
    idx = np.random.choice(len(ts), n_anomalies, replace=False)
    labels = np.zeros(len(ts))

    for i in idx:
        ts[i] += magnitude * np.random.choice([-1, 1])
        labels[i] = 1

    return ts, labels

def sp_encoding(timeseries: np.ndarray):
    mean = np.mean(timeseries)
    scaled_timeseries = timeseries - mean
    scale_factor = 1.0
    if np.any(np.abs(scaled_timeseries) >= 3.0):
        scale_factor = np.max(np.abs(scaled_timeseries)) / 3.0
        scaled_timeseries /= scale_factor
    # meta-prompt
    prompt = f"[Value Offset: {mean:.4f}|Value Scaling: {scale_factor:.4f}]"
    # Stack with structural cue (1.0)
    result_timeseries = np.stack([scaled_timeseries, np.ones_like(scaled_timeseries)], axis=-1).reshape(-1, 1)
    return result_timeseries, prompt, {'offset': float(mean), 'scale_factor': float(scale_factor)}


def generate_trend_freq_shift_ts(n_points=500,change_points=(150, 320),noise_std=0.1,seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_points)
    # Define regimes from change_points
    cps = list(change_points)
    cps = [0] + cps + [n_points]
    n_regimes = len(cps) - 1
    # Example: manually or randomly assign slopes and frequencies per regime
    slopes = [0.02, -0.1, 0.04]          # trend slopes per regime
    freqs  = [0.02, 0.08, 0.03]           # frequencies (cycles / step)
    amps   = [1.0, 0.6, 1.5]              # amplitudes

    y = np.zeros(n_points)
    meta = []

    current_level = 0.0
    current_phase = 0.0

    for k in range(n_regimes):
        start = cps[k]
        end = cps[k + 1]
        idx = np.arange(end - start)
        m = np.random.choice(slopes)
        f = np.random.choice(freqs)
        A = np.random.choice(amps)
        # Trend: linear with slope m, continuous at start
        trend = current_level + m * idx
        # Oscillation: frequency f, continuous phase
        phase = current_phase + 2 * np.pi * f * idx
        osc = A * np.sin(phase)
        y[start:end] = trend + osc
        # Update continuity conditions for next regime
        current_level = trend[-1]  # last trend value at end of regime
        current_phase = phase[-1]  # last phase

        meta.append(
            {
                "regime": k,
                "start": int(start),
                "end": int(end),
                "slope": float(m),
                "freq": float(f),
                "amplitude": float(A),
            }
        )

    # Add noise
    y_noisy = y + rng.normal(0, noise_std, size=n_points)
    print(type(y_noisy))
    return y_noisy


def preprocess_data(sample):
    ### 1. Normalize the timeseries data case of univariate/multivariate time series
    ### 2. Add meta-prompt before each <ts><ts/> token in the input prompt
    new_ts_data=[]
    processed_prompt=[]
    input_prompt=sample['prompt']
    ts_data=sample['timeseries']
    list_patterns = re.split(pattern, input_prompt)
    len_ts_data=len(ts_data)
    ##assert len(list_patterns)-1 == len_ts_data, "Number of time series segments and data do not match."
    normalized_ts, meta_prompt, _ = sp_encoding(ts_data)
    new_ts_data.append(normalized_ts)
    processed_prompt.append(f'{list_patterns[0]}{meta_prompt}{pattern}')
    processed_prompt.append(list_patterns[-1])
                
    sample['prompt'] = "".join(processed_prompt)
    sample['timeseries'] = [ts.tolist() for ts in new_ts_data]
    ##sample['output']=sample['output']
    ####append the last segment after the final <ts><ts/>
    return sample
### test the synthetic pipelin
sample_data=dict()
##raw_ts_data = synthetic_ts(850,0.5,100,50,1.5) ##math function to generate the synthetic data
raw_ts_data=generate_trend_freq_shift_ts(n_points=675,change_points=(78,150),noise_std=0.02,seed=0)
ts_data,label=inject_point_anomalies(raw_ts_data, n_anomalies=5, magnitude=10)
print(type(ts_data))
sample_data['timeseries']=ts_data
sample_data['prompt']="""<|system|>You are a time series analyst<|end|> 
<|user|>The following timeseries data reports the'sales' of company collected over 675 timesteps <ts><ts/>.Generate a summary on the timeseries data in terms of noise ,trend and periodicty<|end|><|assistant|><|thought|>"""

processed_sample=preprocess_data(sample_data)
##print(processed_sample)


import matplotlib.pyplot as plt
plt.plot(ts_data)
plt.show()

##pass these inputs to the 
prompt = processed_sample['prompt']
timeseries=processed_sample['timeseries']

import torch
import torch.nn.functional as F

class TSInferencePreprocessor:
    def __init__(self, tokenizer, patch_len=256, stride=256, device='cuda'):
        self.tokenizer = tokenizer
        self.p = patch_len
        self.s = stride
        self.device = device
        # Fixed constants from your training collate_fn
        self.max_n_per_batch = 10
        self.max_channel_dim = 20
        self.pattern = torch.tensor([0.0, 1.0], device=device)

    def padding_stride(self, ts_list, max_size):
        """Reused from your dataset class"""
        x = torch.tensor(ts_list, dtype=torch.float32, device=self.device).view(1, -1)
        l = x.shape[1]
        
        # Local padding if shorter than patch
        if l < self.p:
            pad_local = (self.p - l) // 2
            num_pad = self.pattern.repeat(pad_local).view(1, -1)
            x_pad = torch.cat([x, num_pad], dim=1)
        else:
            x_pad = x.clone()

        # Global padding/unfolding logic
        padding_0 = max_size - x_pad.shape[1]
        r = (l - self.p) % self.s
        
        if r == 0:
            pad_width = padding_0
            num_repeats = pad_width // 2
            pad = self.pattern.repeat(num_repeats).view(1, -1)
            x_padded = torch.cat([x_pad, pad], dim=1)
            return x_padded[:, :max_size].unfold(1, self.p, self.s)
        else:
            pad_width = (self.s - r) + padding_0
            num_repeats = pad_width // 2
            pad = self.pattern.repeat(num_repeats).view(1, -1)
            x_padded = torch.cat([x_pad, pad], dim=1)
            return x_padded.unfold(1, self.p, self.s)
            
    def generate_ts_mask(self, actual_channels):
        # Simplified version of your mask() function for BS=1
        mask = torch.arange(self.max_channel_dim, device=self.device) < actual_channels
        # (1, max_C, max_N, embed_dim)
        return mask.view(1, -1, 1, 1).expand(-1, -1, self.max_n_per_batch, 3072)
    
    def assemble_attn_mask(self,ts_pairs:torch.tensor,ts_mask,c_in,total_tokens_count,max_N):
        ##displace the (start,end) based on inserted #tokens
        ##print(ts_pairs.shape)
        c_in_tensor=torch.arange(c_in).view(-1,1)
        ts_pair_tensor=ts_pairs
        new_ts_pair=ts_pair_tensor+(c_in_tensor*max_N)
        new_ts_pair[:,1] += max_N
        ## new indices based on the displaced indices
        local_indices= torch.arange(max_N).repeat(c_in, 1)
        new_starts = new_ts_pair[:,0] + 1
        final_ts_indices = ((new_starts.unsqueeze(1)) + local_indices).view(-1)
        assert torch.max(final_ts_indices)<=total_tokens_count,f'Max_of_ts_index:{torch.max(final_ts_indices)},token:{total_tokens_count}'
        ## to get the text_mask tokens
        ##input_tokens =torch.arange(total_tokens_count,dtype=torch.long)
        is_ts_new=torch.zeros(total_tokens_count, dtype=torch.bool)
        is_ts_new[final_ts_indices]=True
        new_text_indices = torch.nonzero(~is_ts_new).squeeze()
        ##ts_mask=torch.ones(new_text_indices.shape,dtype=torch.long) ### the shape textual tokens (inferred input+output_ids)
        attention_mask_container=torch.zeros(total_tokens_count,dtype=torch.long) ## final shape of input_ids+ output_ids + ts_tokens inferred (c_in*max_N+(input_ids+ outputids))
        attention_mask = attention_mask_container.scatter(0,final_ts_indices,ts_mask.flatten())
        textual_mask=torch.ones(new_text_indices.shape,dtype=torch.long)
        attention_mask=attention_mask.scatter(0,new_text_indices,textual_mask)
        return attention_mask

    def preprocess(self, prompt:str, timeseries):
        """
        prompt: str
        timeseries: list of lists (channels)
        """
        # 1. Handle Text
        tokenized = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        input_ids = tokenized['input_ids'].to(self.device)
        
        # 2. Extract TS pairs (reusing your logic)
        ts_start_token = self.tokenizer.convert_tokens_to_ids('<ts>')
        ts_end_token = self.tokenizer.convert_tokens_to_ids('<ts/>')
        
        # Simple extraction for single sample
        stack, ts_pairs = [], []
        for i, tid in enumerate(input_ids[0]):
            if tid == ts_start_token: stack.append(i)
            elif stack and tid == ts_end_token: ts_pairs.append((stack.pop(0), i))

        # 3. Handle Time Series Patching
        len_ts_data = [len(ch) for ch in timeseries]
        max_len = max(len_ts_data)
        num_channels = len(timeseries)
        
        ts_inputs = []
        ts_mask_local=[]
        patch_mask_list=[]
        for i in range(num_channels):
            patched = self.padding_stride(timeseries[i], max_len).squeeze(0) # (N, P)
            ts_inputs.append(patched)

        # 4. Patch Level Padding (to max_N)
        max_N = max(p.shape[0] for p in ts_inputs)
        ch_padded = []
        ##treat univariate as general case
        for ch_ts in ts_inputs:
            curr_N = ch_ts.shape[0]
            ##condition the N<max_N add the padding patches and corresponding attention mask
            if curr_N < self.max_n_per_batch:
                pad_needed = self.max_n_per_batch - curr_N
                ts_padding_len = self.p * pad_needed
                pad = self.pattern.repeat(ts_padding_len // 2).view(pad_needed, self.p)
                ch_padded.append(torch.cat([ch_ts, pad], dim=0))
                patch_mask =torch.cat([torch.ones(curr_N,dtype=torch.long),torch.zeros(pad_needed,dtype=torch.long)])
                patch_mask_list.append(patch_mask)
                ts_mask_local=torch.stack(patch_mask_list,dim=0)
                textual_tokens=input_ids[0].shape[0]
                ts_tokens=(10*num_channels)
                total_tokens=textual_tokens+ts_tokens
                attn_mask=self.assemble_attn_mask(torch.tensor(ts_pairs),ts_mask_local,num_channels,total_tokens,self.max_n_per_batch)
                #call the assemble attention mask
            else:
                ch_padded.append(ch_ts[:self.max_n_per_batch]) ##clipping to the max_N allowed
                attn_mask=torch.ones(self.max_n_per_batch+input_ids[0].shape[0],dtype=torch.long)
            
        

        ##assemble_attn_mask based on ts_pair

        # 5. Channel Dimension Padding (to max_channel_dim)
        # Shape: (max_N, num_channels, P)
        x = torch.stack(ch_padded, dim=1) 
        pad_channel = self.max_channel_dim - num_channels
        # F.pad format: (left, right, top, bottom) for last two dims
        ts_padded_channel = F.pad(x, (0, 0, 0, pad_channel), "constant", 0.0)
        ts_padded_channel = ts_padded_channel.unsqueeze(0) # Add batch dim: (1, max_N, max_C, P)

        # 6. Generate Masks (Matching your collate_fn)
        # ts_mask for the embedding layer
        ts_token_mask = self.generate_ts_mask(num_channels)
        # ch_mask for the attention layer
        ch_mask = (torch.arange(self.max_channel_dim, device=self.device) < num_channels).unsqueeze(0)

        return {
            'input_ids': input_ids,
            'time_series_padded': ts_padded_channel,
            'ts_mask': ts_token_mask,
            'ch_mask': ch_mask,
            'ts_pairs': torch.tensor([ts_pairs], device=self.device),
             'attn_mask':torch.stack([attn_mask]).to(self.device)}
## get the preprocessing pipeline
## custom class
pre_processor=TSInferencePreprocessor(tokenizer_modified, patch_len=256, stride=256, device=device)

print(pre_processor.preprocess(prompt,timeseries)['time_series_padded'].shape)##of size (b,n,c,p)

print(pre_processor.preprocess(prompt,timeseries)['attn_mask'])

print(pre_processor.preprocess(prompt,timeseries)['ch_mask'])


print(pre_processor.preprocess(prompt,timeseries)['ts_mask'].device) ### b,c,N,P

pre_processor.preprocess(prompt,timeseries)['ts_pairs'].shape

###ts_encoder architecture
import torch
import torch.nn as nn  
class ts_encoder_mlp(nn.Module):
    def __init__(self,max_patches,max_channel,patch_length,d_model,device=None):
        super(ts_encoder_mlp,self).__init__()
        self.max_patches=max_patches
        self.max_channel=max_channel
        self.d_model=d_model
        self.d_ff=2*d_model
        self.patch_length=patch_length
        ##self.shared_embeding=shared_embeding
        self.pe_feature=patch_length
        self.meta_feature_ts=8
        self.meta_feature_ch=3
        self.device=device
        
        ## positional encoding for the local timesteps/patch_len and the channel's dimension
        self.ts_pos=nn.Embedding(self.pe_feature,self.meta_feature_ts)
        self.ch_pos = nn.Embedding(self.max_channel+1,self.meta_feature_ch,padding_idx=self.max_channel)

        self.W_p=nn.Sequential(
            nn.Linear((self.patch_length+self.patch_length*self.meta_feature_ts+self.patch_length*self.meta_feature_ch),self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff,self.d_model)
        )
    
    def forward(self,x,ch_mask):
        bs,N,max_ch,p = x.shape
        x_reshaped=x.unsqueeze(-1)  ## (bs,N,c_in,p,1)
        ts_pos_embeds=self.ts_pos(torch.arange(p).to(x.device))
        ts_pos_embeds=ts_pos_embeds.expand(bs,N,max_ch,p,self.meta_feature_ts)

        ##filtered_ch_idx=self.filter_ch_idx(bs,c_in)
        idx = torch.arange(self.max_channel).expand(bs,-1).view(bs,-1)
        filtered_idx = torch.where(ch_mask,idx.to(self.device), self.max_channel)
        ch_pos_embeds=self.ch_pos(filtered_idx)
        ch_pos_embeds=ch_pos_embeds.unsqueeze(1).unsqueeze(-2)  ## [1,1,c_in,1,ts_embed]
        ch_pos_embeds=ch_pos_embeds.expand(-1,N,max_ch,p,self.meta_feature_ch)
        ##print(ch_pos_embeds.shape)
        
        ts_plus_embed = torch.cat([x_reshaped, ts_pos_embeds, ch_pos_embeds], dim=-1)
        ##print(ts_plus_embed.shape)
        x_reshaped = ts_plus_embed.view(bs,N,self.max_channel,-1)
        
        z = self.W_p(x_reshaped)
        ##print(f'z.shape before return: {z.shape}')
        return z.view(bs,max_ch,N,-1)  ## (bs,N,c_in,d_model)

### wrapper for the inference engine
##load the base ll_model and send it as param
## parameters of input prompt: str and the raw timeseries data
class MultiModalInferenceEngine:
    def __init__(self,prompt,raw_ts,base_model,tokenizer,checkpoint_dir,device=device):
        self.prompt=prompt
        self.raw_ts=raw_ts
        self.device = device
        self.cfg={"max_patches": 10,   # Example: your time-series feature size
            "max_channel": 20,         # Example: MLP hidden layer size
            "patch_length": 256,       # Must match Phi-4's hidden_size
            "d_model": 3072,}          # Example: number of MLP layers
        
        self.tokenizer = tokenizer
        ##self.ts_token_id = self.tokenizer.convert_tokens_to_ids("<ts>")
        
        # 2. Load Base LLM and Resize
        self.base_model=base_model
        """base_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-4-mini", 
            torch_dtype=torch.float16
        )"""
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        # 3. Load PEFT Adapters
        self.model = PeftModel.from_pretrained(self.base_model, f"{checkpoint_dir}/phi4-ts-adapter_3")
        self.model.to(self.device).eval()
        
        # 4. Initialize and Load TS Encoder
        self.ts_encoder = ts_encoder_mlp(
            max_patches=self.cfg['max_patches'],
            max_channel=self.cfg['max_channel'],
            patch_length=self.cfg['patch_length'],
            d_model=self.cfg['d_model'],
            device=self.device
        )
        # Loading from the state_dict saved during training
        self.ts_encoder.load_state_dict(torch.load(f"{checkpoint_dir}/ts_encoder_stage3.pth"))
        self.ts_encoder.to(self.device).eval()

    @torch.no_grad()
    def predict(self,pre_processor:TSInferencePreprocessor, max_new_tokens=100):
        """
        text_query: "The signal is <ts> <ts/>. What is the trend?"
        padded_ts_input: List of tensors, each (max_patches, patch_length)
        """
        # --- Preprocessed data object---
        # Initialize padded tensor (BS=1, N, Max_Ch, P)
        ts_input=pre_processor.preprocess(self.prompt,self.raw_ts)['time_series_padded']
        ch_mask=pre_processor.preprocess(self.prompt,self.raw_ts)['ch_mask']
        ts_mask=pre_processor.preprocess(self.prompt,self.raw_ts)['ts_mask']
        attn_mask=pre_processor.preprocess(self.prompt,self.raw_ts)['attn_mask']
        ##ts_input = torch.zeros(1, self.cfg['max_patches'], self.cfg['max_channel'], self.cfg['patch_length']).to(self.device)
        ###ch_mask = torch.zeros(1, self.cfg['max_channel'], dtype=torch.bool).to(self.device)
        
        """for i, ts_tensor in enumerate(raw_ts_list):
            if i < self.cfg['max_channel']:
                ts_input[0, :, i, :] = ts_tensor
                ch_mask[0, i] = True"""
        
        # Encode TS
        # ts_embedding output: (bs, max_ch, max_patches, d_model)
        ts_embedding = self.ts_encoder(ts_input, ch_mask)
        ##print(f'ts_embedding_shape:{ts_embedding.shape}')
        # Slicing like training: only keep the active channels
        ts_embedding_sliced = ts_embedding[ts_mask] # Flattened tokens

        # --- Tokenizing Text ---
        ##inputs = self.tokenizer(text_query, return_tensors="pt").to(self.device)
        input_ids=pre_processor.preprocess(self.prompt,self.raw_ts)['input_ids']
        
        # --- Dynamic ts_pairs Generation ---
        # Find indices of <ts> tags in the prompt
        ##indices = (inputs.input_ids[0] == self.ts_token_id).nonzero(as_tuple=True)[0]
        # Map each active channel to a <ts> placeholder
        ##ts_pairs = torch.stack([indices, indices], dim=1).unsqueeze(0) # (1, num_active_ch, 2)
        ts_pairs=pre_processor.preprocess(self.prompt,self.raw_ts)['ts_pairs']

        # --- Assemble Embeddings ---
        # Use refined assembly logic (handling the 10 tokens per channel)
        input_embeds = self._assemble_inference_embeds(
            input_ids, ts_embedding_sliced, ts_pairs)

        # --- Generation ---
        output_ids = self.model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            do_sample=False,
            ##num_beams=3,
            temperature=0.1
        )
        
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _assemble_inference_embeds(self, input_ids, ts_embeddings, ts_pairs):
        # Implementation based on our previous scatter logic
        bs = input_ids.shape[0]
        num_ts_tokens = 10 # Hardcoded to match training view
        text_emb_dim = self.cfg['d_model']
        
        input_embeds = self.model.get_input_embeddings()(input_ids)
        channels = ts_pairs.shape[1]
        ts_embeddings = ts_embeddings.view(bs, channels, num_ts_tokens, text_emb_dim)
        
        # For simplicity in inference, we assume BS=1
        curr_ts_embeds = ts_embeddings[0].view(-1, text_emb_dim)
        curr_text_embeds = input_embeds[0]
        curr_ts_pairs = ts_pairs[0]

        T_new = curr_text_embeds.shape[0] + (channels * num_ts_tokens)
        
        # Scatter indices logic
        local_indices = torch.arange(num_ts_tokens, device=self.device).repeat(channels, 1)
        new_starts = curr_ts_pairs[:, 0] + 1
        final_ts_indices = (new_starts.unsqueeze(1) + local_indices).view(-1)
        is_ts_new = torch.zeros(T_new, dtype=torch.bool, device=self.device)
        is_ts_new[final_ts_indices] = True
        new_text_indices = torch.nonzero(~is_ts_new).squeeze()

        final_embeds = torch.zeros((T_new, text_emb_dim), device=self.device, dtype=input_embeds.dtype)
        final_embeds.index_copy_(0, new_text_indices, curr_text_embeds)
        final_embeds.index_copy_(0, final_ts_indices, curr_ts_embeds)
        ##print(f'input_embed_shape:{final_embeds.shape}')
        
        return final_embeds.unsqueeze(0)

###call the Inference engine and pass the parameters
import os
import os
import json
checkpoint_dir=os.getcwd()
model_name="/home/mmk/projects/def-zonata/mmk/hf_cache/hub/models--microsoft--Phi-4-mini-reasoning/snapshots/7a8c4e2e81eae20a606d811f475d7dc316dd916a"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
##model_name='microsoft/Phi-4-mini-reasoning'
model = AutoModelForCausalLM.from_pretrained(model_name,local_files_only=True,trust_remote_code=True)
##redundant tokenizer
tokenizer =AutoTokenizer.from_pretrained(model_name,local_files_only=True)

engine = MultiModalInferenceEngine(prompt,timeseries,model,tokenizer_modified,checkpoint_dir=file_path,device=device)

response=engine.predict(pre_processor, max_new_tokens=500)
print(response)





