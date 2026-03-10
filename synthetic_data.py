
###synthetic data generation 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import re
import torch
import orjson
import json
pattern ="<ts><ts/>"
json_file="./synthetic_data.jsonl"
###create class to generate the sample of synthetis ts_data and constant prompt
class synthetics_data_generator:
    def __init__(self,timesteps,sample_size,prompt):
        super().__init__()
        self.timesteps=timesteps
        self.sample_size=sample_size
        self.prompt=prompt
    
    def sp_encoding(self,timeseries:np.ndarray):
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
        
    def generate_trend_freq_shift_ts(self,n_points=500,change_points=(100,110,280),noise_std=0.1,seed=0):
        rng = np.random.default_rng(seed)
        t = np.arange(n_points)
        # Define regimes from change_points
        cps = list(change_points)
        cps = [0] + cps + [n_points]
        n_regimes = len(cps) - 1
        # Example: manually or randomly assign slopes and frequencies per regime
        slopes = [0.02, -0.1, 0.04]         # trend slopes per regime
        freqs  = [0.02, 0.08, 0.03]         # frequencies (cycles / step)
        amps   = [1.0, 0.6, 1.5]            # amplitudes

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
       
    def inject_point_anomalies(self,ts:np.array, n_anomalies=3, magnitude=15.0):
        ts = ts.copy()
        idx = np.random.choice(len(ts), n_anomalies, replace=False)
        labels = np.zeros(len(ts))

        for i in idx:
            ts[i] += magnitude * np.random.choice([-1, 1])
            labels[i] = 1

        return ts, labels  
    
    def process_ts_data(self,sample):
        new_ts_data=[]
        processed_prompt=[]
        input_prompt=sample['prompt']
        ts_data=sample['timeseries']
        
        list_patterns = re.split(pattern, input_prompt)
        len_ts_data=len(ts_data)
        ##assert len(list_patterns)-1 == len_ts_data, "Number of time series segments and data do not match."
        normalized_ts, meta_prompt, _ =self.sp_encoding(ts_data)
        new_ts_data.append(normalized_ts)
        processed_prompt.append(f'{list_patterns[0]}{meta_prompt}{pattern}')
        processed_prompt.append(list_patterns[-1])
                    
        sample['prompt'] = "".join(processed_prompt)
        sample['timeseries'] = [ts.tolist() for ts in new_ts_data]
        print(sample)
        return sample
    
    def process_sample(self):
        dataset =[]
        sample =dict()
        for i in range(self.sample_size):
            ts_data=self.generate_trend_freq_shift_ts(n_points=128*4,change_points=(22,60),noise_std=0.1,seed=0)
            ts_data_1,_=self.inject_point_anomalies(ts_data, n_anomalies=3, magnitude=15.0)
            sample['timeseries']=ts_data_1
            sample['prompt']=self.prompt
            sample=self.process_ts_data(sample)
            
            dataset.append(sample)
            
        print(len(dataset))
        return dataset
            
    def write_jsonl(self,dataset:list,file):
        with open(file, "w",encoding='utf-8') as f:
            for obj in dataset:
                f.write(json.dumps(obj))
                f.write("\n")
            
prompt="""|system|>You are a time series analyst<|end|> 
<|user|>The following timeseries data reports the'sales' of company collected over period of time <ts><ts/>.Generate a summary on the timeseries data in terms of noise ,trend and periodicty<|end|><|assistant|><|thought|>"""

gen=synthetics_data_generator(512,5,prompt)      
data = gen.process_sample()
gen.write_jsonl(data,json_file)

"""
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

##final return value as with the structual cue [x_1,1.0,x_2,1.0,....]
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
    return sample,normalized_ts
### test the synthetic pipeline

sample_data=dict()
##raw_ts_data = synthetic_ts(850,0.5,100,50,1.5) ##math function to generate the synthetic data
raw_ts_data=generate_trend_freq_shift_ts(n_points=128*4,change_points=(4,77),noise_std=0.02,seed=0)
ts_data,label=inject_point_anomalies(raw_ts_data, n_anomalies=5, magnitude=10)
print(type(ts_data))

sample_data['timeseries']=ts_data
sample_data['prompt']=<|system|>You are a time series analyst<|end|> 
<|user|>The following timeseries data reports the'sales' of company collected over period of time <ts><ts/>.Generate a summary on the timeseries data in terms of noise ,trend and periodicty<|end|><|assistant|><|thought|>

processed_sample,ts_norm=preprocess_data(sample_data)

plt.plot(ts_data)
plt.show()"""

###create a constant seq_len timeseries data

