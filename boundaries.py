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

from generate import ByteTokenizer, load_from_pretrained

import os

def generate(
    model,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.9,
):
    """Generate text from the model, yielding tokens as they're generated.

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

    inference_cache = model.allocate_inference_cache(
        1, input_ids.shape[1] + max_tokens, dtype=torch.bfloat16
    )

    with torch.inference_mode():
        mask = torch.ones(input_ids.shape, device=device, dtype=torch.bool)
        output = model.forward(input_ids, mask=mask, inference_params=inference_cache)

    logits = output.logits[0, -1, :] / temperature

    for _ in range(max_tokens):
        # Apply top-p sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        if next_token.item() == tokenizer.eos_idx:
            break

        current_token = next_token.unsqueeze(0)
        yield current_token

        with torch.inference_mode():
            output = model.step(current_token, inference_cache)

        # Get logits and apply temperature
        logits = output.logits[0, -1, :] / temperature


def evaluate_boundaries(
    model,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
):
    device = next(model.parameters()).device
    tokenizer = ByteTokenizer()
    total_boundaries = 0
    total_generated_tokens = 0

    for prompt in prompts:
        encoded = tokenizer.encode([prompt], add_bos=True)[0]
        input_ids = torch.tensor(
            encoded["input_ids"], dtype=torch.long, device=device
        ).unsqueeze(0)

        inference_cache = model.allocate_inference_cache(
            1, input_ids.shape[1] + max_tokens, dtype=torch.bfloat16
        )

        with torch.inference_mode():
            mask = torch.ones(input_ids.shape, device=device, dtype=torch.bool)
            output = model.forward(input_ids, mask=mask, inference_params=inference_cache)

        generated_tokens = 0
        boundary_count = 0
        logits = output.logits[0, -1, :] / temperature

        for _ in range(max_tokens):
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -float("inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            if next_token.item() == tokenizer.eos_idx:
                break

            current_token = next_token.unsqueeze(0)
            generated_tokens += 1

            with torch.inference_mode():
                output = model.step(current_token, inference_cache)

            # Count boundaries from all levels
            step_boundaries = sum(
                bp.boundary_mask.sum().item() for bp in output.bpred_output
            )
            boundary_count += step_boundaries

            logits = output.logits[0, -1, :] / temperature

        total_generated_tokens += generated_tokens
        total_boundaries += boundary_count

    avg_boundaries_per_token = (
        total_boundaries / total_generated_tokens if total_generated_tokens > 0 else 0
    )
    avg_chunk_size = (
        total_generated_tokens / total_boundaries if total_boundaries > 0 else 0
    )

    return {
        "total_boundaries": total_boundaries,
        "total_generated_tokens": total_generated_tokens,
        "avg_boundaries_per_token": avg_boundaries_per_token,
        "avg_chunk_size": avg_chunk_size,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate boundary tokens from an H-Net model")
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
        "--corpus",
        type=str,
        required=True,
        help="Path to the text corpus file",
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=256,
        help="Length of each prompt from the corpus (default: 256)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=50,
        help="Number of prompts to evaluate (default: 50)",
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

    # print model architecture
    print(model)

    # print number of model parameters
    print(sum(p.numel() for p in model.parameters()))

    # Load corpus
    try:
        with open(args.corpus, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"Error loading corpus: {e}")
        sys.exit(1)

    # Split into prompts
    prompts = []
    start = 0
    while start + args.prompt_length < len(text) and len(prompts) < args.num_prompts:
        prompts.append(text[start : start + args.prompt_length])
        start += args.prompt_length

    if not prompts:
        print("No prompts could be generated from the corpus.")
        sys.exit(0)

    print(f"Evaluating {len(prompts)} prompts from the corpus...")

    results = evaluate_boundaries(
        model,
        prompts,
        args.max_tokens,
        args.temperature,
        args.top_p,
    )

    print("\nEvaluation Results:")
    print(f"Total generated tokens: {results['total_generated_tokens']}")
    print(f"Total boundary tokens: {results['total_boundaries']}")
    print(f"Average boundaries per token: {results['avg_boundaries_per_token']:.4f}")
    print(f"Average chunk size: {results['avg_chunk_size']:.4f}")

if __name__ == "__main__":
    main()
