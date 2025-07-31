import numpy as np
import json
import torch
import argparse
import collections
import types
try:
    from tqdm import tqdm
except ImportError:  # fallback if tqdm not available
    def tqdm(x, *args, **kwargs):
        return x

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

    return encoded['input_ids'], output.bpred_output



def prefill_batch(model, prompts):
    """Prefill the model on a batch of prompts (<=32) and return encoded lists and
    routing module outputs.

    Returns
    -------
    encoded_list : list[list[int]]
        Token IDs for each prompt (with BOS included).
    rm_output : list[RoutingModuleOutput]
        Same list returned by the model; each element contains a `boundary_mask`
        of shape (B, L).
    """
    assert len(prompts) <= 32, "Batch size capped at 32 to fit GPU memory"
    device = next(model.parameters()).device

    tokenizer = ByteTokenizer()
    encoded_structs = tokenizer.encode(prompts, add_bos=True)
    encoded_list = [e["input_ids"] for e in encoded_structs]

    max_len = max(len(ids) for ids in encoded_list)
    pad_id = tokenizer.eos_idx  # use EOS as a harmless padding value

    input_ids = torch.full((len(prompts), max_len), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)

    for i, ids in enumerate(encoded_list):
        cur_len = len(ids)
        input_ids[i, :cur_len] = torch.tensor(ids, dtype=torch.long, device=device)
        mask[i, :cur_len] = True

    with torch.inference_mode():
        output = model.forward(input_ids, mask=mask)

    return encoded_list, output.bpred_output


def boundary_print(encoded, rm_output, tokenizer):
    """
    Print the prompt with formatting based on boundary predictions

    Args:
        encoded: Encoded input text prompt
        rm_output: Boundary predictions
    """

    staged_boundary = [stage.boundary_mask[:, 1:] for stage in rm_output]
    
    # Align stages by padding with None to maintain consistent indexing
    aligned_staged_boundary = []
    if staged_boundary:
        # First stage keeps its original form
        aligned_staged_boundary.append(staged_boundary[0])
        
        # For subsequent stages, expand back to original size
        for i in range(1, len(staged_boundary)):
            prev_stage = aligned_staged_boundary[i-1][0]  # Get the previous aligned stage
            current_stage = staged_boundary[i][0]  # Current stage values
            
            # Create aligned version by inserting None where prev_stage was False
            aligned_stage = []
            current_idx = 0
            
            for j in range(len(prev_stage)):
                if prev_stage[j]:  # Previous stage was True
                    if current_idx < len(current_stage):
                        aligned_stage.append(current_stage[current_idx])
                        current_idx += 1
                    else:
                        aligned_stage.append(None)  # Safety fallback
                else:  # Previous stage was False
                    aligned_stage.append(None)
            
            aligned_staged_boundary.append([aligned_stage])
    
    staged_boundary = aligned_staged_boundary

    # strip the bos token
    encoded = encoded[1:]

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
    encoded_idx = 0
    while len(encoded) > 0:
        boundary_buf = []
        res = ""
        decoded = 0
        # Try to decode progressively to handle UTF-8 characters properly
        for j in range(1, 4):
            try:
                res = tokenizer.decode(encoded[:j])    
                decoded = j
                break
            except:
                pass
        
        # Collect multi-stage boundary information and boundary_buf for the tokens we're about to consume
        stage_info_list = []
        boundary_buf = []
        for token_offset in range(decoded):
            current_token_idx = encoded_idx + token_offset
            
            # Collect stage info for this token
            stage_info = []
            for stage_idx in range(len(staged_boundary)):
                if current_token_idx < len(staged_boundary[stage_idx][0]):
                    stage_val = staged_boundary[stage_idx][0][current_token_idx]
                    stage_info.append(stage_val)
                else:
                    stage_info.append(None)
            
            stage_info_list.append(stage_info)
            
            # Use stage 0 for boundary_buf
            boundary_val = stage_info[0] if len(stage_info) > 0 and stage_info[0] is not None else False
            boundary_buf.append(boundary_val)

        # --------------------------------------------------------------------------------
        # Classification logic for this UTF-8 character
        # --------------------------------------------------------------------------------

        def classify(vals):
            """Classify a list of boolean boundary values into one of four categories."""
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
        if len(staged_boundary) > 1:
            for info in stage_info_list:
                s1 = info[0] if len(info) > 0 and info[0] is not None else False
                if s1:  # token survived stage-1
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

        # Print with appropriate colouring
        if color_code:
            print(f"{color_code}{res}\033[0m", end="", flush=True)
        else:
            print(f"{res}", end="", flush=True)

        encoded_idx += decoded
        
        encoded = encoded[decoded:]
    print("\n")



# ----------------------------------------------------------------------------------
# Analytics utilities
# ----------------------------------------------------------------------------------

def _wrap_stage_for_sample(rm_output, sample_idx):
    """Extract a single-sample view of every stage's boundary mask."""
    return [types.SimpleNamespace(boundary_mask=stage.boundary_mask[sample_idx:sample_idx+1]) for stage in rm_output]


def analyze_chunking(model, dataset, num_samples=256, batch_size=32, num_examples=3):
    """Compute chunk length and boundary statistics on *num_samples* prompts.

    The function prints a concise textual summary and also shows *num_examples*
    colourised prompts using `boundary_print` to give a qualitative sense of
    the model's chunking behaviour.
    """
    tokenizer = ByteTokenizer()
    len_stats = collections.Counter()
    pair_stats = collections.Counter()

    # --- new statistics containers ---
    token_total = np.zeros(256, dtype=int)
    token_boundary = np.zeros(256, dtype=int)
    char_total = collections.Counter()
    char_boundary = collections.Counter()

    processed = 0
    example_prompts = []
    example_encodings = []
    example_rm = None

    while processed < num_samples:
        batch_prompts = []
        for _ in range(min(batch_size, num_samples - processed)):
            batch_prompts.append(dataset[processed]["question"].strip())
            processed += 1
        encoded_list, rm_output = prefill_batch(model, batch_prompts)

        # stage-0 boundaries, drop BOS
        stage0_mask = rm_output[0].boundary_mask[:, 1:].cpu().numpy()

        for i, (enc, boundary_row) in enumerate(zip(encoded_list, stage0_mask)):
            seq_len = len(enc) - 1  # without BOS
            true_idx = np.where(boundary_row)[0]
            n_chunks = true_idx.size + 1
            pair_stats[(n_chunks, seq_len)] += 1

            last = 0
            if true_idx.size == 0:
                len_stats[seq_len] += 1
            else:
                for s in true_idx:
                    chunk_len = s - last + 1
                    len_stats[chunk_len] += 1
                    last = s + 1
                if last < seq_len:
                    len_stats[seq_len - last] += 1

            # --- per-token & per-character boundary stats ---
            tokens_wo_bos = enc[1:]
            char_buf = []
            for idx_token, tok in enumerate(tokens_wo_bos):
                is_bdry = bool(boundary_row[idx_token])
                token_total[tok] += 1
                if is_bdry:
                    token_boundary[tok] += 1
                char_buf.append(tok)
                try:
                    ch = bytes(char_buf).decode("utf-8")
                    char_total[ch] += 1
                    if is_bdry:
                        char_boundary[ch] += 1
                    char_buf = []
                except UnicodeDecodeError:
                    pass

        # store first batch's rm_output for qualitative display
        if len(example_prompts) < num_examples:
            take = min(num_examples - len(example_prompts), len(batch_prompts))
            example_prompts.extend(batch_prompts[:take])
            example_encodings.extend(encoded_list[:take])
            if example_rm is None:
                example_rm = rm_output  # keep full batch; we'll slice later

    # --- textual summary ---


    all_lengths = [l for l, c in len_stats.items() for _ in range(c)]
    if all_lengths:
        arr = np.array(all_lengths)
        print(f"Chunk length  mean={arr.mean():.2f}, median={np.median(arr)}, std={arr.std():.2f}")

        # --- byte token boundary frequency ---
        print("\nByte-token boundary frequency (sorted by boundary %, desc):")
        tok_stats = []
        for tok_id in range(256):
            tot = token_total[tok_id]
            if tot == 0:
                continue
            bd = token_boundary[tok_id]
            pct = 100.0 * bd / tot
            tok_stats.append((pct, tok_id, bd, tot))
        for pct, tok_id, bd, tot in sorted(tok_stats, key=lambda x: x[0], reverse=True):
            print(f"  token {tok_id:3d}: {bd}/{tot} ({pct:.1f}%)")

        # --- character boundary frequency (top 100 by boundary %) ---
        print("\nCharacter boundary frequency (top 100 by boundary %):")
        char_stats = []
        for ch, tot in char_total.items():
            bd = char_boundary.get(ch, 0)
            pct = 100.0 * bd / tot
            char_stats.append((pct, tot, ch, bd))
        char_stats_sorted = sorted(char_stats, key=lambda x: x[0], reverse=True)[:100]
        for pct, tot, ch, bd in char_stats_sorted:
            printable = repr(ch)[1:-1]
            print(f"  '{printable}': {bd}/{tot} ({pct:.1f}%)")

    # --- qualitative examples ---
    print("\n----- Example prompts with coloured boundaries (stage-0/1) -----")
    for idx, (enc, text) in enumerate(zip(example_encodings, example_prompts)):
        print(f"\nExample {idx+1}:\n")
        stages_for_sample = _wrap_stage_for_sample(example_rm, idx)
        boundary_print(enc, stages_for_sample, tokenizer)


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
    parser.add_argument(
        "--analyze-samples",
        type=int,
        default=0,
        help="If >0, run chunking analysis over this many GSM8K training samples then exit",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=3,
        help="Number of example prompts to visualise during analysis",
    )

    args = parser.parse_args()

    print("Loading model...")
    try:
        model = load_from_pretrained(args.model_path, args.config_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    dataset = load_dataset("gsm8k", "main")
    print(dataset)

    # If user requested analysis, run it and exit
    if args.analyze_samples > 0:
        ds_split = dataset["train"] if isinstance(dataset, dict) and "train" in dataset else dataset
        analyze_chunking(model, ds_split, num_samples=args.analyze_samples, batch_size=32, num_examples=args.examples)
        return

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
        boundary_print(input_ids, rm_output, tokenizer)
        print("Done!")


if __name__ == "__main__":
    main()
