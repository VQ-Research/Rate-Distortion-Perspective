import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import einsum
from einops import rearrange
from torch import distributed as tdist

class BSQ(nn.Module):
    def __init__(self, args):
        super(BSQ, self).__init__()
        self.args = args
        self.D = args.project_dim
        self.L = args.L
        self.codebook_size = self.L ** self.D
        self.register_buffer('basis', 2 ** torch.arange(self.D-1, -1, -1))
        self.scale = 1. / (self.D ** 0.5)

    def quantize(self, z):
        zhat = torch.where(z > 0, torch.tensor(1, dtype=z.dtype, device=z.device), torch.tensor(-1, dtype=z.dtype, device=z.device))
        return z + (zhat - z).detach()

    def codes_to_indices(self, zhat):
        return ((zhat + 1) / 2 * self.basis).sum(axis=-1).to(torch.int32)

    def forward(self, z_enc):
        B, C, H, W = z_enc.shape
        z = rearrange(z_enc, 'b c h w -> b h w c') 
        z_flat = z.reshape(-1, C).contiguous() 
        zhat = self.quantize(z_flat)
        token = self.codes_to_indices(zhat.detach())
        
        zhat = zhat * self.scale
        z_dec = zhat.view(z.shape).permute(0, 3, 1, 2).contiguous()
        commit_loss = self.args.beta * F.mse_loss(z_dec.detach(), z_enc)

        histogram = token.bincount(minlength=self.codebook_size).float()
        handler = tdist.all_reduce(histogram, async_op=True)
        handler.wait()

        codebook_usage_counts = (histogram > 0).float().sum()
        utilization = codebook_usage_counts.item() / self.codebook_size
            
        avg_probs = histogram/histogram.sum(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        loss = commit_loss
        return z_dec, loss, utilization, perplexity

    def collect_eval_info(self, z_enc):
        B, C, H, W = z_enc.shape
        z = rearrange(z_enc, 'b c h w -> b h w c') 
        z_flat = z.reshape(-1, C).contiguous() 
        zhat = self.quantize(z_flat)
        token = self.codes_to_indices(zhat.detach())
        
        zhat = zhat * self.scale
        z_dec = zhat.view(z.shape).permute(0, 3, 1, 2).contiguous()

        histogram = token.bincount(minlength=self.codebook_size).float()
        handler = tdist.all_reduce(histogram, async_op=True)
        handler.wait()
        return z_dec, histogram

    def collect_reconstruction(self, z_enc):
        B, C, H, W = z_enc.shape
        z = rearrange(z_enc, 'b c h w -> b h w c') 
        z_flat = z.reshape(-1, C).contiguous() 
        zhat = self.quantize(z_flat)
        token = self.codes_to_indices(zhat.detach())
        
        zhat = zhat * self.scale
        z_dec = zhat.view(z.shape).permute(0, 3, 1, 2).contiguous()
        return z_dec