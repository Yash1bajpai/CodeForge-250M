import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return (self.weight * x_normed).to(input_dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = float(base)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class CodeForgeAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = int(config["hidden_size"])
        self.num_heads = int(config["num_attention_heads"])
        self.num_kv_heads = int(config.get("num_key_value_heads", self.num_heads))
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=int(config.get("max_position_embeddings", 2048)),
            base=float(config.get("rope_theta", 10000.0))
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(v, seq_len=q_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # PyTorch 2.0+ Scaled Dot Product Attention (SDPA / FlashAttention-2)
        # Prevents 13GB attention matrix VRAM spike! Keeps VRAM usage under ~5 GB!
        if attention_mask is not None and attention_mask.dim() == 4:
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        else:
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=(q_len > 1))
            
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)

class CodeForgeMLP(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = int(config["hidden_size"])
        self.intermediate_size = int(config["intermediate_size"])
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class CodeForgeDecoderLayer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = int(config["hidden_size"])
        self.self_attn = CodeForgeAttention(config)
        self.mlp = CodeForgeMLP(config)
        self.input_layernorm = RMSNorm(self.hidden_size, eps=float(config.get("rms_norm_eps", 1e-6)))
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=float(config.get("rms_norm_eps", 1e-6)))

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class CodeForgeModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.vocab_size = int(config["vocab_size"])
        self.hidden_size = int(config["hidden_size"])
        self.num_layers = int(config["num_hidden_layers"])
        
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = nn.ModuleList([CodeForgeDecoderLayer(config) for _ in range(self.num_layers)])
        self.norm = RMSNorm(self.hidden_size, eps=float(config.get("rms_norm_eps", 1e-6)))
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        if config.get("tie_word_embeddings", False):
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        bsz, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=None)  # SDPA handles causal masking natively!
            
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            
        return logits, loss

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
