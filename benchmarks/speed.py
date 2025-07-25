import numpy as np
import json
import torch
import argparse
import sys
from omegaconf import ListConfig
import time
import timeit

from hnet.models.mixer_seq import HNetForCausalLM
from hnet.models.config_hnet import (
    AttnConfig,
    SSMConfig,
    HNetConfig,
)

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

    return model


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

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs dtype: {input_ids.dtype}")

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


def main():
    parser = argparse.ArgumentParser(description="Generate text from an H-Net model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt file)",
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
    print("DEBUG: Entered main()", file=sys.stderr, flush=True)

    print("Loading model...")

    configs = ['configs/hnet_2stage_XL_fused.json', 'configs/hnet_2stage_XL.json']

    tokenizer = ByteTokenizer()

    with open("./prompts/speed_prompt.txt", "r") as f:
        prompt = f.read()

    print(f"Prompt length: {len(prompt)} characters", file=sys.stderr)
    print(f"Prompt preview: {prompt[:100]}...", file=sys.stderr)

    for config in configs:
        try:
            model = load_from_pretrained(args.model_path, config)
            print(f"Loading model {config}...")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

        num_iters = 0

        ttft = []
        throughput = []
        while num_iters < 10:
            num_iters += 1
            prompt = "tell me about large language models"

            if not prompt:
                continue

            print(
                f"\nGenerating (max_tokens={args.max_tokens}, temperature={args.temperature}, top_p={args.top_p})"
            )

            #print(f"\033[92m{prompt}\033[0m", end="")
            # Start timing for throughput and time-to-first-token
            start_time = time.perf_counter()
            first_token_measured = False
            print("\033[92m[Debug] Timer started\033[0m", file=sys.stderr, flush=True)
            token_count = 0
            buf = []

            for token in generate(
                model,
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            ):
                buf.append(token)
                token_count += 1
                # Measure time to first token
                if not first_token_measured:
                    time_to_first = time.perf_counter() - start_time
                    print(f"\n\033[91mTime to first token: {time_to_first*1000:.5f} ms\033[0m", file=sys.stderr, flush=True)
                    ttft.append(time_to_first)
                    first_token_measured = True

                decoded = None
                res = None
                for j in range(1, min(len(buf), 4)):
                    try:
                        res = tokenizer.decode(buf[:j])
                        decoded = j
                    except:
                        pass

                if res is not None:
                    #print(res, end="", flush=True)
                    buf = buf[decoded:]

            # Compute and display overall throughput
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            throughput_tokens_per_second = token_count / elapsed if elapsed > 0 else float('inf')
            print(f"\n\033[91mThroughput: {throughput_tokens_per_second:.5f} tokens/sec (elapsed: {elapsed*1000:.5f} ms)\033[0m", file=sys.stderr, flush=True)
            throughput.append(throughput_tokens_per_second)

        print(f"\033[94mAverage time to first token: {np.mean(ttft[1:])*1000:.5f} ms\033[0m", file=sys.stderr, flush=True)
        print(f"\033[94mAverage throughput: {np.mean(throughput[1:]):.5f} tokens/sec\033[0m", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
