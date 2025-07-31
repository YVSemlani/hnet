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

    # Allocate inference cache
    inference_cache = model.allocate_inference_cache(
        1, input_ids.shape[1] + max_tokens, dtype=torch.bfloat16
    )

    # Forward pass
    with torch.inference_mode():
        mask = torch.ones(input_ids.shape, device=device, dtype=torch.bool)
        output = model.forward(input_ids, mask=mask, inference_params=inference_cache)

    # set initial boundary indicator
    #print("PREFILLED BOUNDARY INDICATORS")
    #print(f"output.bpred_output: {output.bpred_output}")
    boundary_indicators = [stage.boundary_mask[:, -1] for stage in output.bpred_output]
    #print(f"boundary_indicators: {boundary_indicators}")

    # Generate tokens
    logits = output.logits[0, -1, :] / temperature

    # Autoregressive generation
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
        yield current_token, boundary_indicators # yield current token and status as a boundary token

        with torch.inference_mode():
            output = model.step(current_token, inference_cache)

        # update boundary indicator
        #print(f"output.bpred_output: {output.bpred_output}")
        boundary_indicators = [stage.boundary_mask.item() for stage in output.bpred_output]
        #print(f"boundary_indicators: {boundary_indicators}")

        # Get logits and apply temperature
        logits = output.logits[0, -1, :] / temperature

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

    print("Color legend:")
    print("  \033[101mRed background\033[0m: Passed stage 1 but not stage 2 (all boundary)")
    print("  \033[104mBlue background\033[0m: Passed stage 1 but not stage 2 (most boundary)")
    print("  \033[103mYellow background\033[0m: Passed stage 1 but not stage 2 (most non-boundary)")
    print("  \033[100mGray background\033[0m: Passed stage 1 but not stage 2 (no boundary)")
    print("  \033[1;91mBold Red\033[0m: All tokens are boundary tokens (stage 1)")
    print("  \033[94mBlue\033[0m: Most tokens are boundary tokens (stage 1)")
    print("  \033[93mYellow\033[0m: Most tokens are not boundary tokens (stage 1)")
    print("  Normal: No boundary tokens (stage 1)")
    print()

    while True:
        print("\n")
        prompt = input("\nPrompt: ").strip()

        if not prompt:
            continue

        print(
            f"\nGenerating (max_tokens={args.max_tokens}, temperature={args.temperature}, top_p={args.top_p})"
        )

        print(f"\033[92m{prompt}\033[0m\n", end="")
        token_count = 0
        buf = []
        boundary_buf = []

        for token, boundary_indicator in generate(
            model,
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        ):
            buf.append(token)
            token_count += 1
            boundary_buf.append(boundary_indicator)

            decoded = None
            res = None
            for j in range(1, min(len(buf), 4)):
                try:
                    res = tokenizer.decode(buf[:j])
                    decoded = j
                except:
                    pass

            if res is not None:
                # Two-stage colouring (stage-0 foreground, stage-1 background) –
                # logic mirrors hnet.gsm8k.boundary_print
                stage_info_list = boundary_buf[:decoded]

                def classify(vals):
                    if not vals:
                        return "none"
                    true_count = sum(vals)
                    total = len(vals)
                    if true_count == total:
                        return "all"
                    elif true_count == 0:
                        return "none"
                    elif true_count > total / 2:
                        return "most"
                    else:
                        return "most_not"

                # Stage-1 classification (foreground colours)
                stage1_vals = [info[0] if len(info) > 0 and info[0] is not None else False for info in stage_info_list]
                class1 = classify(stage1_vals)

                # Stage-2 classification (background colours) – only for tokens that were boundary in stage-1
                stage2_vals = []
                if any(len(info) > 1 for info in stage_info_list):
                    for info in stage_info_list:
                        s1 = info[0] if len(info) > 0 and info[0] is not None else False
                        if s1:
                            s2 = info[1] if len(info) > 1 and info[1] is not None else False
                            stage2_vals.append(s2)

                fg_map = {
                    "all": "\033[1;91m",
                    "most": "\033[94m",
                    "most_not": "\033[93m",
                    "none": "",
                }
                bg_map = {
                    "all": "\033[101m",
                    "most": "\033[104m",
                    "most_not": "\033[103m",
                    "none": "\033[100m",
                }

                if stage2_vals:
                    class2 = classify(stage2_vals)
                    color_code = bg_map[class2]
                else:
                    color_code = fg_map[class1]

                if color_code:
                    print(f"{color_code}{res}\033[0m", end="", flush=True)
                else:
                    print(f"{res}", end="", flush=True)

                buf = buf[decoded:]
                boundary_buf = boundary_buf[decoded:]


if __name__ == "__main__":
    main()
