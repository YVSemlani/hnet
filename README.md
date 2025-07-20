# Efficient Inference for HNet

This repo contains my improvements to the H-Net architecture for efficient inference.

## About

I plan to keep consistency with the original H-Net repository and the code is organized very similarly to the original repository
```
configs/
hnet/
├── benchmarks/        # Directory for benchmarks
|   └── prefill.py     (benchmarking script comparing prefill performance of different model configurations)
|   └── routing.py     (benchmarking script for only routing module fused and un-fused)
|   └── speed.py       (benchmarking script for TTFT and throughput in generation)
├── models/            # Directory for H-Net
|   ├── config_hnet.py     (defines the config for the H-Net)
|   ├── hnet.py            (h-net as a (B, L, D) -> (B, L, D) sequence model)
│   └── mixer_seq.py       (wrapper to turn h-net into a language model)
└── modules/           # Directory of model components
    ├── dc.py              (modeling code for the dynamic chunking mechanism)
    └── isotropic.py       (code for isotropic, i.e. non-hierarchical components)
    └── ops/
        └── fused_dc.py   (fused cosine similarity, probability, and boundary mask computation)
├── hf/                # Directory for pretrained models
├── reports/           # Directory for results of various experiments
generate.py        # Script for inference/generation
```

## Installation

I elect to use UV to manage my dependencies. Use -vn when installing to keep track of long install (flash-attn, mamba, & causal-conv1d) progress.

```
cd hnet
uv venv
uv pip install -e . -vn # or uv sync (not sure if this works)
```

## Usage

### Benchmarks

Benchmarks are currently hardcoded to use the hnet_2stage_XL model. Configs have been edited to indicate whether to use fused or un-fused dynamic chunking.

```
uv run speed.py --model-path hf/hnet_2stage_XL.pt
uv run prefill.py
uv run routing.py
```

### Generation
```
uv run generate.py --model-path hf/hnet_2stage_XL.pt --config-path configs/hnet_2stage_XL.json --max-tokens 1024 --temperature 1.0 --top-p 1.0
```

## Development

I'd like to implement the following and will add more as I think of optimizations:

- [ ] Full fused dynamic chunking module

## Citations

### H-Net

```
@article{hnet,
  title={Dynamic Chunking for End-to-End Hierarchical Sequence Modeling},
  author={Hwang, Sukjun and Wang, Brandon and Gu, Albert},
  journal={arXiv preprint arXiv:2507.07955},
  year={2025}
}
```