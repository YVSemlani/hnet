import numpy as np
import json
import torch
import argparse
import sys
from omegaconf import ListConfig

from hnet.models.mixer_seq import HNetForCausalLM
from hnet.models.config_hnet import (
    AttnConfig,
    SSMConfig,
    HNetConfig,
)

from hnet.utils.tokenizers import ByteTokenizer
from hnet.utils.loaders import load_from_pretrained

from datasets import load_dataset

def prefill(
    model,
    prompt: str,
):
    """Prefill the model with the prompt to extract boundaries

    Args:
        model: HNetForCausalLM model
        prompt: Input text prompt
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Top-p sampling parameter

    Yields:
        Generated text token by token as strings
    """
    device = next(model.parameters()).device
    tokenizer = ByteTokenizer()

    # Tokenize prompt
    encoded = tokenizer.encode([prompt], add_bos=True)[0]
    input_ids = torch.tensor(
        encoded["input_ids"], dtype=torch.long, device=device
    ).unsqueeze(0)


    # Forward pass
    with torch.inference_mode():
        mask = torch.ones(input_ids.shape, device=device, dtype=torch.bool)
        output = model.forward(input_ids, mask=mask)

    return input_ids, output.bpred_output

def boundary_print(encoded, rm_output):
    """
    Print the prompt with formatting based on boundary predictions

    Args:
        encoded: Encoded input text prompt
        rm_output: Boundary predictions
    """

    staged_boundary = [stage.boundary_mask for stage in rm_output]

    # map a boundary indicator to each byte token
    prompt_map = [] # list of lists with each index corresponding to that index input_id in encoded

    for idx, token in enumerate(encoded):
        token_map = []
        for stage in staged_boundary:
            try:
                token_map.append(stage[idx])
            except:
                token_map.append(None)
        prompt_map.append(token_map)

    # pretty print the prompt
    encoded_idx = 0
    while len(encoded) > 0:
        boundary_buf = []
        res = ""
        decoded = 0
        for j in range(1, 4):
            boundary_buf.append(prompt_map[encoded_idx][0]) # only take stage 0 for now
            try:
                res = tokenizer.decode(encoded[:j])
                decoded = j
            except:
                pass
        if False not in boundary_buf[:decoded]: # all byte level tokens are boundary tokens
            print(f"\033[1;91m{res}\033[0m", end="", flush=True)
        elif True not in boundary_buf[:decoded]: # all byte level tokens are not boundary tokens
            print(f"{res}", end="", flush=True)
        elif sum(boundary_buf[:decoded]) > len(boundary_buf[:decoded]) / 2: # most byte level tokens are boundary tokens
            print(f"\033[94m{res}\033[0m", end="", flush=True)
        else: # most byte level tokens are not boundary tokens
            print(f"\033[93m{res}\033[0m", end="", flush=True)
        
        encoded = encoded[decoded:]
        encoded_idx += decoded



def main():
    parser = argparse.ArgumentParser(description="Generate text from an H-Net model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the model configuration (.json file)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate (default: 1024)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter (default: 1.0)",
    )
    args = parser.parse_args()

    print("Loading model...")
    try:
        model = load_from_pretrained(args.model_path, args.config_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    tokenizer = ByteTokenizer()

    while True:
        print("\n")
        prompt = input("\nPrompt: ").strip()

        if not prompt:
            continue

        print(
            f"\nGenerating (max_tokens={args.max_tokens}, temperature={args.temperature}, top_p={args.top_p})"
        )

        print(f"\033[92m{prompt}\033[0m\n", end="")

        print("Prefilling...")
        input_ids, rm_output = prefill(model, prompt)
        print("Printing boundaries...")
        boundary_print(input_ids, rm_output)
        print("Done!")


if __name__ == "__main__":
    main()
