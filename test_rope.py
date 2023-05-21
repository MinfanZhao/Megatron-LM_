# test rope


import importlib.util
import torch

from torch import einsum, nn

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        # print('inv', inv_freq)
        self.register_buffer('inv_freq', inv_freq)
        if importlib.util.find_spec('einops') is None:
            raise RuntimeError("einops is required for Rotary Embedding")

    def forward(self, max_seq_len, offset=0):
        seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.stack([freqs, freqs],dim=-1).flatten(-2)
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
    x = torch.cat((-x2, x1), dim=-1)
    return x


def _rotate_half_0(x):
    from einops import rearrange
    
    # x1, x2 = x.unbind(dim=-2)
    print(x[0], x.shape)
    x = x.reshape(*x.shape[:-1], -1, 2).transpose(-1,-2)
    x = x.reshape(*x.shape[:-2], -1)
    return x

def _rotate_half_1(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    from einops import rearrange
    
    # x1, x2 = x.unbind(dim=-2)
    # print(x[0], x.shape)
    x = x.reshape(*x.shape[:-1], -1, 2).transpose(-1,-2)
    x = x.reshape(*x.shape[:-2], -1)
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    x = torch.cat((-x2, x1), dim=-1)
    return x

def _rotate_half_llama(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    from einops import rearrange
    
    x = rearrange(x, '... (j d) -> ... j d', d=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-2).transpose(-1,-2).flatten(-2)
    return x


def apply_rotary_pos_emb(t, freqs):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    rot_dim = freqs.shape[-1]
    # print(t.shape, freqs.shape)
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    
    t = (t * freqs.cos()) + (_rotate_half(t) * freqs.sin())
    # t = (t * freqs.cos()) + (t * freqs.sin())
    return torch.cat((t, t_pass), dim=-1)


def apply_rotary_pos_emb_llama_legacy(t, freqs):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    rot_dim = freqs.shape[-1]
    # print(t.shape, freqs.shape)
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    
    t = (_rotate_half_0(t) * freqs.cos()) + (_rotate_half_1(t) * freqs.sin())
    # t = (t * freqs.cos()) + (t * freqs.sin())
    t = torch.cat((t, t_pass), dim=-1)
    # print(t)
    t = t.reshape(*t.shape[:-1], 2, -1).transpose(-1,-2)
    t = t.reshape(*t.shape[:-2], -1)    
    return t


def apply_rotary_pos_emb_llama(t, freqs):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    rot_dim = freqs.shape[-1]
    # print(t.shape, freqs.shape)
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * freqs.cos()) + (_rotate_half_llama(t) * freqs.sin())
    t = torch.cat((t, t_pass), dim=-1)  
    return t


from typing import Optional, Tuple

def llama_apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    print(xq.shape, xq_.shape, (xq_ * freqs_cis).shape, torch.view_as_real(xq_ * freqs_cis).shape)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    # print("shape", freqs_cis.shape,x.shape[1], x.shape[-1])
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # print(freqs)
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

if __name__ == "__main__":
    s, b, h = 4, 3, 6
    q,k = torch.arange(0,int(s*b*h)).reshape(s,b,h).float(), torch.arange(1,1+int(s*b*h)).reshape(s,b,h).float()
    megatron_rope = RotaryEmbedding(h)
    megatron_rope_value = megatron_rope(s)
    # print(megatron_rope_value)
    if isinstance(megatron_rope_value, tuple):
        megatron_rope_value = megatron_rope_value
    else:
        megatron_rope_value = ((megatron_rope_value,) * 2)
    
    q_rope, k_rope = megatron_rope_value
    magatron_xq = apply_rotary_pos_emb_llama(q.view(s,b,1,h), q_rope)
    print(f"q_rope:{magatron_xq}")
    # print(q_rope.cos(), q_rope.sin())
    # print(f"k_rope:{apply_rotary_pos_emb(k.view(s,b,1,h), k_rope)}")

    llama_freqs_cis = precompute_freqs_cis(h, 2*s)
    llama_freqs_cis = llama_freqs_cis[:s]
    # print(f"llama freqs:{llama_freqs_cis}")
    q, k = q.transpose(0,1).reshape(b,s,1,h), k.transpose(0,1).reshape(b,s,1,h)
    xq, xk = llama_apply_rotary_emb(q,k,llama_freqs_cis)
    # print(xq.size())
    xq, xk = xq.transpose(0,1), xk.transpose(0,1)
    # print(xq.size())
    
    print(f"llama q_rope: {xq}")
    # print(f"llama k_rope: {xk}")
    
    


