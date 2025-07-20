# benchmark end2end generation speed with and without fused dc

import torch
import torch.nn.functional as F

import triton
import triton.language as tl
from triton.runtime import driver

import json 

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

import matplotlib.pyplot as plt

DEVICE = triton.runtime.driver.active.get_active_torch_device()
torch.set_float32_matmul_precision('high')

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

def benchmark(model, input_ids, mask, inference_cache, warmup=3, iters=10):
    # run a warmup
    for _ in range(warmup):
        inference_cache = model.allocate_inference_cache( # inference cache must be reset between forward calls otherwise model treats next call like a step rather than prefill
            1, input_ids.shape[1] + MAX_TOKENS, dtype=torch.bfloat16
        )
        output = model.forward(input_ids, mask=mask, inference_params=inference_cache)
    torch.cuda.synchronize()

    total_ms = 0.0
    
    # run the benchmark
    for _ in range(iters):
        inference_cache = model.allocate_inference_cache( # inference cache must be reset between forward calls otherwise model treats next call like a step rather than prefill
            1, input_ids.shape[1] + MAX_TOKENS, dtype=torch.bfloat16
        )
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        output = model.forward(input_ids, mask=mask, inference_params=inference_cache)
        end_event.record()
        end_event.synchronize()
        total_ms += start_event.elapsed_time(end_event)
        del inference_cache
        del output
        gc.collect()
    return total_ms / iters

if __name__ == "__main__":
    results_path = "reports/prefill.png"
    model_path = "hf/hnet_2stage_XL.pt"
    model_configs = ["configs/hnet_2stage_XL_fused.json", "configs/hnet_2stage_XL.json"]
    MODELS = []
    MAX_TOKENS = 1024
    results = {}
    for model_config in model_configs:
        model = load_from_pretrained(model_path, model_config)
        #print(model)
        MODELS.append(model)
        results[model_config] = []
        print(f"Loaded model {model_config}")


    # create dummy input with different batch_sizes
    batch_sizes = [1] # need to get CausalLM to support batched inputs
    seq_lens = [1024, 2048, 4096, 8192, 16384]
    inputs = []
    masks = []
    inference_caches = []
    #for batch_size in batch_sizes:
    for seq_len in seq_lens:
        print(f"\nCreating dummy input for seq len {seq_len}")
        input_ids = torch.randint(0, 256, (1, seq_len), device=DEVICE, dtype=torch.int64)
        mask = torch.ones(1, seq_len, device=DEVICE, dtype=torch.bool)
        for model_idx, model in enumerate(MODELS):
            inference_cache = model.allocate_inference_cache(
                1, input_ids.shape[1] + MAX_TOKENS, dtype=torch.bfloat16
            )
            avg_ms = benchmark(model, input_ids, mask, inference_cache)
            print(f"{model_configs[model_idx]} prefill for {seq_len} tokens took {avg_ms:.6f} ms")
            results[model_configs[model_idx]].append(avg_ms)
        # cleanup
        del input_ids
        del mask
        del inference_cache
        torch.cuda.empty_cache()
        gc.collect()

    # plot results
    for model_config, times in results.items():
        plt.plot(seq_lens, times, label=model_config)
    plt.xlabel("Sequence Length")
    plt.ylabel("Average Prefill Time w/ 3 warmup & 10 iters (ms)")
    plt.legend()
    plt.savefig(results_path)