import json
import torch
from omegaconf import ListConfig

from hnet.models.mixer_seq import HNetForCausalLM
from hnet.models.config_hnet import (
    AttnConfig,
    SSMConfig,
    HNetConfig,
)

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
