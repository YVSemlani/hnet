# benchmark end2end generation speed with and without fused dc
import torch
import torch.nn.functional as F

import triton
import triton.language as tl
from triton.runtime import driver

import json
import os 

from hnet.models.mixer_seq import HNetForCausalLM
from hnet.models.config_hnet import (
    AttnConfig,
    SSMConfig,
    HNetConfig,
)

import gc
from omegaconf import ListConfig
import numpy as np
import time
import itertools

import matplotlib.pyplot as plt
import pandas as pd

DEVICE = triton.runtime.driver.active.get_active_torch_device()
#torch.set_float32_matmul_precision('high')

class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.bos_idx = 254
        self.eos_idx = 255
        self.dtype = np.uint8

    def encode(self, seqs, add_bos=False, add_eos=False, **kwargs):
        total_outputs = []
        for text in seqs:
            text_byte = text.encode("utf-8")

            if add_bos:
                text_byte = bytes([self.bos_idx]) + text_byte
            if add_eos:
                text_byte = text_byte + bytes([self.eos_idx])
            text_byte = bytearray(text_byte)
            text_byte_ids = np.array(text_byte, dtype=self.dtype)

            total_outputs.append({"input_ids": text_byte_ids})

        return total_outputs

    def decode(self, tokens, **kwargs):
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        return bytearray(tokens).decode("utf-8", **kwargs)

def load_from_pretrained(model_path: str, model_config_path: str):
    """Load model from pretrained checkpoint.

    Args:
        model_path: Path to the model checkpoint (.pt file)
        model_config_path: Path to the model configuration (.json file)

    Returns:
        Loaded HNetForCausalLM model
    """
    # Load configuration
    with open(model_config_path, "r") as f:
        config = json.load(f)

    # Create config objects
    attn_cfg = AttnConfig(**config.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**config.pop("ssm_cfg"))
    hnet_cfg = HNetConfig(**config, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)

    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HNetForCausalLM(hnet_cfg, device=device, dtype=torch.bfloat16)
    model.eval()

    # Load checkpoint
    major, minor = map(int, torch.__version__.split('.')[:2])
    if (major, minor) >= (2, 6):
        with torch.serialization.safe_globals([ListConfig]):
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
    else:
        state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # compile model
    #model = torch.compile(model)

    return model

def benchmark(model, input_ids, mask, warmup=5, iters=50):
    batch_size = input_ids.shape[0]
    # run a warmup
    for _ in range(warmup):
        with torch.no_grad():
            output = model.forward(input_ids, mask=mask)
        del output
        torch.cuda.empty_cache()
    torch.cuda.synchronize()

    total_ms = 0.0
    
    # run the benchmark
    for _ in range(iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.no_grad():
            output = model.forward(input_ids, mask=mask)
        end_event.record()
        end_event.synchronize()
        total_ms += start_event.elapsed_time(end_event)
        del output
        torch.cuda.empty_cache()
        gc.collect()
    return total_ms / iters

if __name__ == "__main__":
    results_dir = "reports/prefill"
    os.makedirs(results_dir, exist_ok=True)
    
    model_path = "hf/hnet_2stage_XL.pt"
    model_configs = ["configs/hnet_2stage_XL_fused.json", "configs/hnet_2stage_XL.json"]
    model_names = ["Fused", "Unfused"]
    MAX_TOKENS = 1024
    
    # Create a dictionary to store results by (batch_size, seq_len) key
    results_dict = {}
    batch_sizes = [1, 2, 4, 8, 16, 32] # CausalLM support batched inputs when inference params are not provided
    seq_lens = [8192]
    num_memory_ops = 2
    HEAD_DIM = 1024
    for model_idx, model_config in enumerate(model_configs):
        model = load_from_pretrained(model_path, model_config)
        print(f"Loaded model {model_config}")
        
        # Test this model with all configurations
        for batch_size, seq_len in itertools.product(batch_sizes, seq_lens):
            print(f"\nCreating dummy input for batch size {batch_size} and seq len {seq_len}")
            input_ids = torch.randint(0, 256, (batch_size, seq_len), device=DEVICE, dtype=torch.int64)
            mask = torch.ones(batch_size, seq_len, device=DEVICE, dtype=torch.bool)
            
            ms = benchmark(model, input_ids, mask)
            
            # Use (batch_size, seq_len) as key to group results
            key = (batch_size, seq_len)
            if key not in results_dict:
                results_dict[key] = {'Batch Size': batch_size, 'Sequence Length': seq_len}
            tbps = lambda ms: num_memory_ops * input_ids.numel() * input_ids.element_size() * HEAD_DIM * 1e-12 / (ms * 1e-3)
            results_dict[key][model_names[model_idx]] = tbps(ms)
            print(f"{model_names[model_idx]} prefill for {batch_size} batch size and {seq_len} tokens took {ms:.6f} ms and achieved {tbps(ms):.6f} TB/s")
            # cleanup
            del input_ids
            del mask
            torch.cuda.empty_cache()
            gc.collect()
        
        # Clean up model
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Convert dictionary to list of results
    results = list(results_dict.values())
    
    # plot results
    # Convert results list to DataFrame first
    df = pd.DataFrame(results)
    
    # Group results by sequence length and plot separately
    seq_lens = sorted(df['Sequence Length'].unique())
    for seq_len in seq_lens:
        plt.figure(figsize=(10, 6))
        seq_data = df[df['Sequence Length'] == seq_len].copy()
        seq_data = seq_data.sort_values('Batch Size')
        
        for model_name in model_names:
            plt.plot(seq_data['Batch Size'], seq_data[model_name], 
                    marker='o', linewidth=2, markersize=8, label=model_name)
            
        plt.title(f"Prefill Time vs Batch Size (SeqLen={seq_len})", fontsize=14)
        plt.xlabel("Batch Size", fontsize=12)
        plt.ylabel("Average Prefill Time (ms)", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/prefill_len{seq_len}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # save results to csv
    df["% Speedup"] = (df["Fused"] - df["Unfused"]) / df["Unfused"] * 100
    df.sort_values(by=['Batch Size', 'Sequence Length'], inplace=True)
    df.to_csv(f"{results_dir}/prefill.csv", index=False)
    
    # Print summary statistics
    print("\n=== Benchmark Results Summary ===")
    print(df.to_string(index=False))
    
    # Calculate and print average speedup
    avg_speedup = df["% Speedup"].mean()
    print(f"\nAverage speedup: {avg_speedup:.2f}%")