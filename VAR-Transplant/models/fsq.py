import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import einsum
from einops import rearrange
from torch import distributed as tdist
from torch import tensor, Tensor, int32

class FSQ(nn.Module):
    def __init__(self, args):
        super(FSQ, self).__init__()
        self.args = args
        self.D = args.project_dim
        self.L = args.L
        self.codebook_size = self.L**self.D

        self.levels = [self.L for i in range(self.D)]
        self.basis = torch.cumprod(tensor([1] + self.levels[:-1]), dim = 0, dtype = int32).cuda()

    def round_ste(self, z):
        """ round with straight through gradients. """
        zhat = z.round()
        return z + (zhat - z).detach()

    def bound(self, z, eps = 1e-8):
        """ Bound `z`, an array of shape (..., d). """
        pre_offset = self.L * torch.ones(self.D, dtype=torch.int32, device=z.device)
        half_l = (pre_offset - 1) * (1 + eps) / 2
        offset = torch.where(pre_offset % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        bounded_z = (z + shift).tanh() * half_l - offset
        half_width = pre_offset // 2
        return self.round_ste(bounded_z)/ half_width
    
    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        half_width = self.L // 2
        zhat =  zhat * half_width + half_width
        return (zhat * self.basis).sum(dim = -1).round().to(int32)

    def forward(self, z_enc):
        B, C, H, W = z_enc.shape
        z = rearrange(z_enc, 'b c h w -> b h w c') 
        z_flat = z.reshape(-1, C).contiguous()  

        zhat = self.bound(z_flat)
        token = self.codes_to_indices(zhat)

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

        zhat = self.bound(z_flat)
        token = self.codes_to_indices(zhat)
        z_dec = zhat.view(z.shape).permute(0, 3, 1, 2).contiguous()

        histogram = token.bincount(minlength=self.codebook_size).float()
        handler = tdist.all_reduce(histogram, async_op=True)
        handler.wait()
        return z_dec, histogram

    def collect_reconstruction(self, z_enc):
        B, C, H, W = z_enc.shape
        z = rearrange(z_enc, 'b c h w -> b h w c') 
        z_flat = z.reshape(-1, C).contiguous()  

        zhat = self.bound(z_flat)
        z_dec = zhat.view(z.shape).permute(0, 3, 1, 2).contiguous()
        return z_dec