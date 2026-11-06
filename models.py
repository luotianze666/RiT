import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from rit import RiT
from typing import Literal, Optional, Tuple
import einops

class FlowMatchingModel(nn.Module):
    def __init__(self, 
                data_channels: int = 3,
                data_size: int = 256,
                patch_size: int = 16,
                hidden_size: int = 768,
                depth: int = 12,
                num_heads: int = 12,
                mlp_ratio: float = 4.0,
                attn_drop: float = 0.0,
                proj_drop: float = 0.0,
                num_classes: int = 1000,
                bottleneck_dim: int = 128,
                in_context_len: int = 32,
                in_context_start: int = 4,
                patch_size_final: int = 8,
                hidden_size_final: int = 384,
                depth_final: int = 2,
                num_heads_final: int = 6,
                decoder_input: bool = True, 
                shortcut_type: str = 'fraction_anchored',
                use_F_sdpa: bool = True,
                use_v_pred: bool = True, 
                principal_dim: int = 0,
                cfg_drop_rate: float = 0.1,
                statistics_path: Optional[str] = None
                ):
        
        
        super().__init__()
        self.data_channels = data_channels
        self.data_size = data_size
        self.num_classes = num_classes
        self.principal_dim = principal_dim
        self.use_v_pred = use_v_pred
        self.cfg_drop_rate = cfg_drop_rate
        self.patch_dim = data_channels * patch_size * patch_size 
        self.patch_size = patch_size

        if self.principal_dim > 0:
            
            assert statistics_path is not None
            stats = torch.load(statistics_path)
            mean, cov = stats['mean'].float(), stats['cov'].float()
            L, V = torch.linalg.eigh(cov)
            S = L.clamp(min = 0).sqrt()
            self.register_buffer('mean', mean, persistent = True)
            self.register_buffer('S', S, persistent = True)
            self.register_buffer('V', V, persistent = True)
            
            assert self.principal_dim <= self.patch_dim
            major_s = torch.ones_like(S)
            major_s[:-self.principal_dim] = 0.0
            major_proj = torch.matmul(V * major_s.unsqueeze(0), V.T)
            self.register_buffer('major_proj', major_proj, persistent = True)

        self.network = RiT(data_channels = data_channels, data_size = data_size,
                        patch_size = patch_size, hidden_size = hidden_size, depth = depth, 
                        num_heads = num_heads, mlp_ratio = mlp_ratio, attn_drop = attn_drop, 
                        proj_drop = proj_drop, num_classes = num_classes,
                        bottleneck_dim = bottleneck_dim, in_context_len = in_context_len, 
                        in_context_start = in_context_start, patch_size_final = patch_size_final, 
                        hidden_size_final = hidden_size_final, depth_final = depth_final, 
                        num_heads_final = num_heads_final, decoder_input = decoder_input, 
                        shortcut_type = shortcut_type, use_F_sdpa = use_F_sdpa) 
    
    def affine_proj(self, images):
        # images: [B, C, H, W]
        images = einops.rearrange(images, 'b c (h h1) (w w1) -> b h w (c h1 w1)', 
                        h1 = self.patch_size, w1 = self.patch_size)
        images = torch.einsum('i j, b h w j -> b h w i', self.major_proj, images - self.mean) + self.mean
        images = einops.rearrange(images, 'b h w (c h1 w1) -> b c (h h1) (w w1)', 
                        h1 = self.patch_size, w1 = self.patch_size)
        return images
        
    def get_training_targets(self, x_0: torch.Tensor, t: torch.Tensor, labels: torch.Tensor, limit = 0.05):
        # data samples x_0: [B, C, H, W], time t: [B]
        B, C, H, W = x_0.shape
        
        x_1 = torch.randn_like(x_0)
        if self.principal_dim > 0:
            x_0 = self.affine_proj(x_0)
            x_1 = self.affine_proj(x_1)

        mask = torch.rand(B, device=labels.device) < self.cfg_drop_rate
        labels = torch.where(mask, self.num_classes, labels)

        x_t = t.view(-1, 1, 1, 1) * x_1 + (1 - t).view(-1, 1, 1, 1) * x_0
        
        if self.use_v_pred == True:
            targets = x_1 - x_0
            coefs = torch.ones_like(t)
        else:
            targets = x_0
            coefs = 1.0 / t.clamp(min = limit).pow(2)
    
        return x_t, targets, coefs.flatten(), labels
    
    @torch.inference_mode
    def predict_velocity(self, x_t: torch.Tensor, t: torch.Tensor, 
                         labels: torch.Tensor, cfg_scale: float, 
                         use_bf16_autocast: bool = True, limit: float = 0.05):
        # images: [B, C, H, W], t: [B], s: [B], output: [B, C, H, W]

        if cfg_scale == 1.0:
            with torch.autocast(device_type = 'cuda', dtype = torch.bfloat16, enabled = use_bf16_autocast):
                raw_pred = self.network(x_t, t, labels).float()
        else:
            x_in = torch.cat([x_t, x_t], dim=0)
            t_in = torch.cat([t, t], dim=0)
            y_uncond = torch.full_like(labels, self.network.num_classes)
            y_in = torch.cat([labels, y_uncond], dim=0)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled = use_bf16_autocast):
                all_pred = self.network(x_in, t_in, y_in).float()
            
            raw_pred_c, raw_pred_u = all_pred.chunk(2, dim=0)
            raw_pred = raw_pred_u + cfg_scale * (raw_pred_c - raw_pred_u)

        if self.use_v_pred == True:
            return raw_pred
        else:
            return (x_t - raw_pred) / t.view(-1, 1, 1, 1).clamp(min = limit)
    
    @torch.inference_mode()
    def inference(self, labels: torch.Tensor, solver: Literal['Euler', 'Heun'] = 'Heun', steps: int = 50,
                cfg_scale: float = 3.0, cfg_interval: Tuple[float, float] = (0.0, 0.89), use_bf16_autocast: bool = True):
        # labels: [B]
        assert solver in ['Euler', 'Heun']
        batch_size = labels.size(0)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=labels.device)
        data_channels, data_size = self.data_channels, self.data_size
        x_t = torch.randn((batch_size, data_channels, data_size, data_size), device = labels.device)       
        
        if self.principal_dim > 0:
            x_t = self.affine_proj(x_t)

        for i in range(steps):
            t, t_next = ts[i], ts[i+1]
            dt = t_next - t  
            
            current_cfg = cfg_scale if (cfg_interval[0] <= t <= cfg_interval[1]) else 1.0
            v = self.predict_velocity(x_t, t * torch.ones(batch_size, 
                    device=labels.device), labels, current_cfg, use_bf16_autocast)
            
            if solver == 'Heun':
                # Final Euler step, avoid data-prediciton's divergence at t=0
                if i == steps - 1:
                    x_t = x_t + v * dt
                    
                else:
                    x_pred = x_t + v * dt
                    if self.principal_dim > 0:
                        x_pred = self.affine_proj(x_pred)

                    next_cfg = cfg_scale if (cfg_interval[0] <= t_next <= cfg_interval[1]) else 1.0
                    v_next = self.predict_velocity(x_pred, t_next * torch.ones(batch_size, 
                                device=labels.device), labels, next_cfg, use_bf16_autocast)
                    
                    x_t = x_t + 0.5 * (v + v_next) * dt
            else:
                x_t = x_t + v * dt
                
            if self.principal_dim > 0:
                x_t = self.affine_proj(x_t)
        
        return x_t

    def forward(self, samples: torch.Tensor, t: torch.Tensor, 
                labels: torch.Tensor, use_bf16_autocast: bool = True):
        # samples: [B, C, H, W], t: [B], labels: [B] or None
        
        batch_size = samples.size(0)
        x_t, targets, coefs, labels = self.get_training_targets(x_0 = samples, t = t, labels = labels)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled = use_bf16_autocast):
            predict = self.network(x_t, t, labels).float()
        
        delta = (predict - targets).view(batch_size, -1)
        loss = (coefs * delta.pow(2).mean(dim = -1)).mean()
        loss_mse = delta.pow(2).mean()
        loss_weight = coefs.mean()
        
        return loss, loss_mse, loss_weight
    