# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# JiT: https://github.com/LTH14/JiT
# --------------------------------------------------------

from math import pi
import torch
from torch import nn
import numpy as np
from einops import rearrange, repeat
from typing import Literal

def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs_h = torch.einsum('..., f -> ... f', t, freqs)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r = 2)

        freqs_w = torch.einsum('..., f -> ... f', t, freqs)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r = 2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim = -1)

        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

    def forward(self, t, start_index = 0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)
        return torch.cat((t_left, t, t_right), dim = -1)


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        num_cls_token = 0
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1)

        if num_cls_token > 0:
            freqs_flat = freqs.view(-1, freqs.shape[-1])  # [N_img, D]
            cos_img = freqs_flat.cos()
            sin_img = freqs_flat.sin()

            # prepend in-context cls token
            N_img, D = cos_img.shape
            cos_pad = torch.ones(num_cls_token, D, dtype=cos_img.dtype, device=cos_img.device)
            sin_pad = torch.zeros(num_cls_token, D, dtype=sin_img.dtype, device=sin_img.device)

            self.register_buffer("freqs_cos", torch.cat([cos_pad, cos_img], dim=0), persistent=False)
            self.register_buffer("freqs_sin", torch.cat([sin_pad, sin_img], dim=0), persistent=False)
            # self.freqs_cos = torch.cat([cos_pad, cos_img], dim=0).cuda()  # [N_cls+N_img, D]
            # self.freqs_sin = torch.cat([sin_pad, sin_img], dim=0).cuda()
        else:
            self.register_buffer("freqs_cos", freqs.cos().view(-1, freqs.shape[-1]), persistent=False)
            self.register_buffer("freqs_sin", freqs.sin().view(-1, freqs.shape[-1]), persistent=False)
            # self.freqs_cos = freqs.cos().view(-1, freqs.shape[-1]).cuda()
            # self.freqs_sin = freqs.sin().view(-1, freqs.shape[-1]).cuda()

    def forward(self, t): return  t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    
class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, bottleneck_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Sequential(
                    nn.Linear(in_chans * (patch_size[0]**2), bottleneck_dim, bias = False),
                    nn.Linear(bottleneck_dim, embed_dim, bias = bias)
                )
     
    def forward(self, x):
        return self.proj(x)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings

def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype, device=query.device)

    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0., use_F_sdpa=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if use_F_sdpa == True:
            self.sdpa = F.scaled_dot_product_attention
        else:
            self.sdpa = scaled_dot_product_attention

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        x = self.sdpa(q, k, v,  dropout_p=self.attn_drop.p if self.training else 0.)
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class Decoder(nn.Module):
    """
    The final layer of RiT.
    """
    def __init__(self, hidden_size, patch_size, hidden_size_final, 
                 patch_size_final, num_heads_final, depth_final,
                 mlp_ratio_final, data_size, data_channels, 
                 decoder_input, shortcut_type, use_F_sdpa):
        
        super().__init__()
    
        assert shortcut_type in ['none', 'basic', 'fraction_free', 'fraction_anchored', 'fraction_anchored_separate']

        self.patch_size = patch_size
        self.patch_size_final = patch_size_final

        self.shortcut_type = shortcut_type   
        self.patch_dim_final = data_channels * (patch_size_final ** 2)
        
        adaLN_ratio = {'none': 2, 'basic': 4, 'fraction_free': 5, 'fraction_anchored': 5, 'fraction_anchored_separate': 6}
        self.adaLN_modulation = nn.Sequential(nn.SiLU(),
            nn.Linear(hidden_size_final, adaLN_ratio[self.shortcut_type] * hidden_size_final, bias = True))
        self.norm = RMSNorm(hidden_size_final)
        
        if decoder_input == True:
            self.x_proj = nn.Linear(self.patch_dim_final, hidden_size_final, bias = True)
        
        if depth_final > 0:
            self.feat_rope_final = VisionRotaryEmbeddingFast(
                dim = (hidden_size_final // num_heads_final) // 2,
                pt_seq_len = data_size // self.patch_size_final,
                num_cls_token = 0
            )
            self.blocks_final = nn.ModuleList([
                RiTBlock(hidden_size_final, num_heads_final, mlp_ratio = mlp_ratio_final,
                attn_drop = 0.0, proj_drop = 0.0, use_F_sdpa = use_F_sdpa)
                for _ in range(depth_final)
            ])
        
        if hidden_size_final != hidden_size or patch_size_final != patch_size:
            self.h_in = nn.Linear(hidden_size, hidden_size_final * 
                                ((patch_size // patch_size_final) ** 2), bias = True)
            
        if hidden_size != hidden_size_final:
            self.c_proj = nn.Linear(hidden_size, hidden_size_final)
            
        if self.shortcut_type == 'none':
            self.h_out = nn.Linear(hidden_size_final, self.patch_dim_final, bias = True)
        else:
            self.x_in = nn.Linear(self.patch_dim_final, hidden_size_final, bias = True)
            self.x_out = nn.Linear(hidden_size_final, self.patch_dim_final, bias = True)
            self.h_out = nn.Linear(hidden_size_final, hidden_size_final, bias = True)
        
    def forward(self, h, c, raw_x_t, t, limit = 0.05):
        # h: [B, L, D], c: [B, D], raw_x_t: [B, 3, H, W], t: [B]
        
        length = int(h.size(1) ** 0.5)
        ratio = self.patch_size // self.patch_size_final
        if hasattr(self, "h_in"):
            h = self.h_in(h) 
            
        h = rearrange(h, 'b (l1 l2) (c p1 p2) -> b (l1 p1 l2 p2) c',
            l1 = length, l2 = length, p1 = ratio, p2 = ratio)
        
        raw_x_t = rearrange(raw_x_t, 'b c (l1 p1) (l2 p2) -> b (l1 l2) (c p1 p2)', 
            p1 = self.patch_size_final, p2 = self.patch_size_final) 
        
        if hasattr(self, "x_proj"):
            h = h + self.x_proj(raw_x_t)
        
        if hasattr(self, "c_proj"):
            c = self.c_proj(c)
        
        if hasattr(self, "blocks_final"):
            for block in self.blocks_final:
                h = block(h, c, self.feat_rope_final)
        
        t = t.reshape(-1, 1, 1)
        c = c.unsqueeze(1)
        if self.shortcut_type != 'none':
        
            if self.shortcut_type == 'basic':
                c1, c2, c3, c4 = self.adaLN_modulation(c).chunk(4, dim = -1)
                raw_x_t = (1.0 + c1) * self.x_in(raw_x_t)
                h = (1.0 + c2) * self.h_out(self.norm(h) * (1.0 + c3) + c4)
                out = self.x_out(raw_x_t + h)

            elif self.shortcut_type == 'fraction_free':
                c1, c2, c3, c4, c5 = self.adaLN_modulation(c).chunk(5, dim = -1)
                coef1 = (1.0 + c1) / (t + c2).clamp(min = limit)
                coef2 = (1.0 + c3) / (t + c2).clamp(min = limit)
                raw_x_t = coef1 * self.x_in(raw_x_t) 
                h = coef2 * self.h_out(self.norm(h) * (1.0 + c4) + c5)
                out = self.x_out(raw_x_t + h)
            
            elif self.shortcut_type == 'fraction_anchored':    
                c1, c2, c3, c4, c5 = self.adaLN_modulation(c).chunk(5, dim = -1)
                coef1 = ((1.0 + (1 - t) * c1) / (t + (1 - t) * c2).clamp(min = limit))
                coef2 = ((1.0 + (1 - t) * c3) / (t + (1 - t) * c2).clamp(min = limit))
                raw_x_t = coef1 * self.x_in(raw_x_t) 
                h = coef2 * self.h_out(self.norm(h) * (1.0 + c4) + c5)
                out = self.x_out(raw_x_t + h)
            
            else: # self.shortcut_type == 'fraction_anchored_separate'    
                c1, c2, c3, c4, c5, c6 = self.adaLN_modulation(c).chunk(6, dim = -1)
                coef1 = ((1.0 + (1 - t) * c1) / (t + (1 - t) * c2).clamp(min = limit))
                coef2 = ((1.0 + (1 - t) * c3) / (t + (1 - t) * c4).clamp(min = limit))
                raw_x_t = coef1 * self.x_in(raw_x_t) 
                h = coef2 * self.h_out(self.norm(h) * (1.0 + c5) + c6)
                out = self.x_out(raw_x_t + h)
        
        else:
            c1, c2 = self.adaLN_modulation(c).chunk(2, dim = -1)
            out = self.h_out(self.norm(h) * (1.0 + c1) + c2)
        
        return rearrange(out, 'b (l1 l2) (c p1 p2) -> b c (l1 p1) (l2 p2)',
                l1 = int(out.size(1) ** 0.5) , l2 = int(out.size(1) ** 0.5), 
                p1 = self.patch_size_final, p2 = self.patch_size_final)                                
        
    def initialize(self):

        if hasattr(self, "blocks_final"):
            for block in self.blocks_final:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0.0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0.0)
                        
        if hasattr(self, "x_in"):
            nn.init.orthogonal_(self.x_in.weight)
            nn.init.constant_(self.x_in.bias, 0.0)
            with torch.no_grad():
                self.x_out.weight.copy_(self.x_in.weight.T)
            nn.init.constant_(self.x_out.bias, 0.0)

        if hasattr(self, "x_proj"):
            nn.init.orthogonal_(self.x_proj.weight)
            nn.init.constant_(self.x_proj.bias, 0.0)      
                     
        nn.init.constant_(self.h_out.weight, 0.0)
        nn.init.constant_(self.h_out.bias, 0.0)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0.0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0.0)
        
class RiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, use_F_sdpa=True):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop, use_F_sdpa=use_F_sdpa)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class RiT(nn.Module):
    """
    Raw-injected Transformer.
    """
    def __init__(
        self,
        data_channels = 3,
        data_size = 256,
        patch_size = 16,
        hidden_size = 1024,
        depth = 24,
        num_heads = 16,
        mlp_ratio = 4.0,
        attn_drop = 0.0,
        proj_drop = 0.0,
        num_classes = 1000,
        bottleneck_dim = 128,
        in_context_len = 32,
        in_context_start = 4,
        patch_size_final = 8,
        hidden_size_final = 384,
        depth_final = 2,
        num_heads_final = 6,
        decoder_input = True, 
        shortcut_type = 'fraction_anchored',
        use_F_sdpa = True    
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = data_channels * patch_size * patch_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes
        
        # time and class embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
    
        # linear embed
        self.x_embedder = BottleneckPatchEmbed(data_size, patch_size, data_channels, bottleneck_dim, hidden_size, bias = True)

        # use fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad = False)

        # in-context cls token
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, in_context_len, hidden_size), requires_grad = True)
            torch.nn.init.normal_(self.in_context_posemb, std = 0.02)

        # rope
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = data_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim = half_head_dim,
            pt_seq_len = hw_seq_len,
            num_cls_token = 0
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim = half_head_dim,
            pt_seq_len = hw_seq_len,
            num_cls_token = self.in_context_len
        )
        
        # transformer
        self.blocks = nn.ModuleList([
            RiTBlock(hidden_size, num_heads, mlp_ratio = mlp_ratio,
                     attn_drop = attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop = proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     use_F_sdpa = use_F_sdpa)
            for i in range(depth)
        ])
        
        # decoder
        self.decoder = Decoder(hidden_size, patch_size, hidden_size_final, patch_size_final, 
                               num_heads_final, depth_final, mlp_ratio, data_size, 
                               data_channels, decoder_input, shortcut_type, use_F_sdpa)
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std = 0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std = 0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std = 0.02)
        
        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            
        # Zero-out output layers:
        self.decoder.initialize()
    
    def forward(self, x, t, y):
        # x: [B, C, H, W], t: [B], y: [B]
        
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb
        
        # forward RiT
        raw_x_t = x
        x = self.x_embedder(rearrange(x , 'b c (h h1) (w w1) -> b (h w) (c h1 w1)', 
                            h1 = self.patch_size, w1 = self.patch_size))
        x += self.pos_embed
        
        for i, block in enumerate(self.blocks):
            # in-context
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens += self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)
            x = block(x, c, self.feat_rope if i < self.in_context_start else self.feat_rope_incontext)
                
        return self.decoder(x[:, self.in_context_len:], c, raw_x_t, t)