# coding=utf-8

# The following code has been taken from https://github.com/NVIDIA/NeMo/blob/ \
# 782b4e1652aaa43c8be390d9db0dc89544afa080/nemo/collections/nlp/modules/ \
# common/megatron/rotary_pos_embedding.py

import importlib.util
import torch

from torch import einsum, nn

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, rope_style="megatron"):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        if importlib.util.find_spec('einops') is None:
            raise RuntimeError("einops is required for Rotary Embedding")
        self.rope_style=rope_style

    def forward(self, max_seq_len, offset=0):
        seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        if self.rope_style == "megatron":
            emb = torch.cat((freqs, freqs), dim=-1)
        elif self.rope_style == "llama":
            emb = torch.stack([freqs, freqs],dim=-1).flatten(-2)
        else:
            raise ValueError("Unknown rope style error.")
        # emb [seq_length, .., dim]
        from einops import rearrange
        return rearrange(emb, 'n d -> n 1 1 d')



def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    from einops import rearrange
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)

def _rotate_half_llama(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    from einops import rearrange
    x = rearrange(x, '... (j d) -> ... j d', d=2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-2).transpose(-1,-2).flatten(-2)

def apply_rotary_pos_emb(t, freqs, rope_style='megatron', position_ids=None):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    
    if position_ids is not None:
        cos_part = freqs.cos().squeeze(2).squeeze(1)
        sin_part = freqs.sin().squeeze(2).squeeze(1)
        cos_part = cos_part[position_ids].transpose(0,1).unsqueeze(2)
        sin_part = sin_part[position_ids].transpose(0,1).unsqueeze(2)
    
    else:
        cos_part = freqs.cos()
        sin_part = freqs.sin()
        
    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    if rope_style == 'megatron':
        t = (t * cos_part) + (_rotate_half(t) * sin_part)
    elif rope_style == 'llama':
        t = (t * cos_part) + (_rotate_half_llama(t) * sin_part)
    else:
        raise ValueError("Unknown rope style error.") 
    return torch.cat((t, t_pass), dim=-1)
    
    

if __name__ == "__main__":
    hidden_size = 4096
    num_attention_heads = 32
    dim = hidden_size // num_attention_heads
    rope = RotaryEmbedding(dim, rope_style='llama')
    
