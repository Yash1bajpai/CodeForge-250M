import math
import torch
import torch.nn as nn
from models.architecture import CodeForgeModel, RMSNorm

def init_model_weights(model: CodeForgeModel, initializer_range: float = 0.02):
    """
    LLaMA-2 Style Stable Weight Initialization.
    1. Linear & Embedding layers: Normal distribution N(0, initializer_range^2) or N(0, 1/sqrt(d)).
    2. Residual projection layers (W_o and W_down): Scaled down by 1/sqrt(2 * num_layers) to prevent initial gradient explosion!
    3. RMSNorm layers: Weights initialized to 1.0.
    """
    num_layers = model.num_layers
    residual_scale = 1.0 / math.sqrt(2.0 * num_layers)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            
            if "o_proj" in name or "down_proj" in name:
                with torch.no_grad():
                    module.weight.mul_(residual_scale)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    print(f"--> [Init] Model weights successfully initialized with LLaMA-2 scaling (Residual Scale: {residual_scale:.4f})")
