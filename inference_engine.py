from modules.ts_encoder_rel_bias import PatchTSTEncoder
from modules.conv_module import ConvFeatureExtraction
from modules.ts_encoder import llm_projection
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer
###from ts_dataloader import ts_textual,collate_func
from ts_dataloader_eval_data import ts_textual,collate_func
from torch.utils.data import Dataset,DataLoader
import torch
import json
import os

##print(torch.randn(1,4))
###inference engine (main file to run the inference phase)
### step similar to the llm_wrapper
 ##load the trained ts_encoder
 ### trained LLM weights
 ### expanded tokenizer
 ##Assemble the input embeds
 ##llm_model.generate() using the input_embeds
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file_path="/home/mmk/projects/def-zonata/mmk/version_2/"
checkpoint_dir="/home/mmk/projects/def-zonata/mmk/version_2/stage_2"

model_path="/home/mmk/projects/def-zonata/mmk/hf_cache/hub/models--microsoft--Phi-4-mini-reasoning/snapshots/7a8c4e2e81eae20a606d811f475d7dc316dd916a"
llm_model_path = os.path.abspath(model_path)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

res_file=os.path.join(os.environ["SLURM_TMPDIR"],'response_batch_1.jsonl')
##sft_file=os.path.join(os.environ["SLURM_TMPDIR"],'synthetic_data.jsonl')

eval_data_set=os.path.join(os.environ["SLURM_TMPDIR"],'dataset_b.jsonl')

tokenizer_path =os.path.join(file_path,'llm_tokenizer')
tokenizer_modified = AutoTokenizer.from_pretrained(tokenizer_path)
###print(device) 

ts_dataset=ts_textual(128,128,tokenizer_modified,eval_data_set,device=device)
ts_loader =DataLoader(ts_dataset,batch_size=1,shuffle=False,collate_fn=lambda b:collate_func(b,tokenizer=tokenizer_modified))

class MultiModalInferenceEngine:
    def __init__(self,output_file,model_path,patch_len,conv_layers,tokenizer,checkpoint_dir=None,device=device):
        """self.prompt=prompt
        self.raw_ts=raw_ts"""
        self.device = device
        self.model_path=model_path
        self.patch_len=patch_len
        self.conv_layers=conv_layers
        self.output_file=output_file
    
        ###load the expanded tokenizer
        self.tokenizer=tokenizer
        ##self.ts_token_id = self.tokenizer.convert_tokens_to_ids("<ts>")
        
        # 2. Load Base LLM and Resize the input_embeddings
        self.base_model=AutoModelForCausalLM.from_pretrained(self.model_path,local_files_only=True)
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        # 3. Load PEFT Adapters
        self.model = PeftModel.from_pretrained(self.base_model, f"{checkpoint_dir}/phi4-ts-adapter_ver2")
        self.model = self.model.merge_and_unload()
        self.model.to(self.device).eval()
        
        # 4. Initialize and Load TS Encoder
        self.ts_transformer=PatchTSTEncoder(patch_len=self.patch_len,n_layers=2,d_model=512,n_heads=4,
                             shared_embedding=False,d_ff=1024,norm='Layer',attn_dropout=0.,dropout=0.1,activation='gelu',store_attn=False,res_attention=False,pre_norm=True,pe='zeros',learn_pe=True,verbose=False)
        
        self.ts_conv_module=ConvFeatureExtraction(conv_layers,dropout=0.1)
        
        ###main ts_encoder
        self.ts_encoder=llm_projection(self.ts_conv_module,64,self.ts_transformer,512,1024,3072)

        # Loading from the state_dict saved during training
        self.ts_encoder.load_state_dict(torch.load(f"{checkpoint_dir}/ts_encoder_ver2_final.pth"),strict=False)
        self.ts_encoder.to(self.device).eval()
        
    @torch.no_grad()
    def predict(self,ts_loader:DataLoader,max_new_tokens=100):
        
        """text_query: "The signal is <ts> <ts/>. What is the trend?"
        padded_ts_input: List of tensors, each (max_patches, patch_length)"""
        
        ###responses =[]
        # --- Preprocessed data object---
        # batch of outputs (BS=5, N, Max_Ch, P)
        with open(self.output_file,'a',encoding='utf-8') as f:
            with torch.no_grad(): 
                for batch in ts_loader:
                    ts_input=batch['time_series'].to(self.device)
                    attn_mask=batch['attention_mask'].to(self.device)
                    input_ids=batch['input_ids'].to(self.device)
                    ts_pairs=batch['ts_pairs']
                    ts_seq_index=batch["ts_indices"]
                    textual_index=batch['textual_indices']
                    # Encode TS
                    # ts_embedding output: (bs, max_ch, max_patches, d_model)
                    ts_embedding = self.ts_encoder(ts_input)
                
                # --- Assemble Embeddings ---
                # Use refined assembly logic (handling the 10 tokens per channel)
                    input_embeds = self.v2_assemble_input_embeds(input_ids,ts_embedding,ts_seq_index,textual_index,ts_pairs)
                    print(f'input_embeds:{input_embeds.shape}')

                # --- Generation of batch of prediced tokens
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
                    
                    responses=self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                    ##print(f'Responsed:{responses}')
                
                    for i, text in enumerate(responses):
                        record={"sample_id": f'sample{i}', 
                                "prediction": text.strip()
                                }
                        # Write as a single line JSON (the 'l' in jsonl)
                        f.write(json.dumps(record) + "\n")
                        print('file_written')
                        
        
    def _assemble_inference_embeds(self, input_ids, ts_embeddings, ts_pairs):
        ###based on the scatter logic to assemble the input_context and ts_embeddings
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
        final_embeds.to(self.device)
        ##print(f'input_embed_shape:{final_embeds.shape}')
        
        return final_embeds.unsqueeze(0)
    
    def v2_assemble_input_embeds(self,input_ids,ts_embeddings,ts_token_idx,text_token_idx,ts_pairs:torch.tensor):
        ###logic to assemble textual and ts_tokens batch-wise
        ###assemb_embed_tensor=[] 
        channels=ts_pairs.shape[1]
        bs=ts_embeddings.shape[0]
        c_in=ts_embeddings.shape[1]
        assert c_in==channels
        num_ts_tokens=ts_embeddings.shape[2]
        ts_emb_dim=ts_embeddings.shape[3]

        ##ts_embeddings=ts_embeddings.view(bs*c_in,num_ts_tokens,-1)        
        input_embeds=self.model.get_input_embeddings()(input_ids) ##[bs,seq_len,d_emb]
        flat_ts_embeddings=ts_embeddings.view(-1,c_in*num_ts_tokens,ts_emb_dim).to(self.device)

        text_emb_dim= input_embeds.shape[2]
        assert (ts_emb_dim==text_emb_dim)
        ###new total_sequence length    
        T_new=ts_token_idx.shape[1]+text_token_idx.shape[1]
        print(T_new)
        final_container =torch.zeros((bs,T_new,text_emb_dim),device=self.device,dtype=ts_embeddings.dtype) ### total_idx,total_idx
        
        flat_text_embeddings=input_embeds
        ##get the indices after the <ts>....<ts/> placeholder is offseted
        ts_indices=ts_token_idx.unsqueeze(-1).expand(-1, -1, text_emb_dim).to(self.device)
        ###ts_indices=ts_indices.expand(-1,text_emb_dim)
        text_indices=text_token_idx.unsqueeze(-1).expand(-1, -1, text_emb_dim).to(self.device)
       ###text_indices=text_indices.expand(-1,text_emb_dim)
       
        final_container.scatter_(dim=1,index=ts_indices,src=flat_ts_embeddings)
        final_container.scatter_(dim=1,index=text_indices,src=flat_text_embeddings)
        """final_tensor=ts_embeds_assemb+text_embeds_assemb
        assemb_embed_tensor.append(final_tensor)"""
        
        return final_container.to(self.device)
 
conv_layers =[(128,5,1),(64,3,1)]
###instantiate inference wrapper passing llm_model location
engine = MultiModalInferenceEngine(res_file,llm_model_path,128,conv_layers,tokenizer_modified,checkpoint_dir=checkpoint_dir,device=device)

## loop around batches to return and generate prediction
engine.predict(ts_loader,max_new_tokens=100)
##save the response
""""
    for i, text in enumerate():
        record = {
            "sample_id": i, 
            "prediction": text.strip()
        }
        # Write as a single line JSON (the 'l' in jsonl)
        f.write(json.dumps(record) + "\n")"""