import os
import sys
import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F
from models.vanilla_quantizer import VanillaVectorQuantizer
from models.ema_quantizer import EMAVectorQuantizer
from models.online_quantizer import OnlineVectorQuantizer
from models.wasserstein_quantizer import WassersteinVectorQuantizer
from models.mmd_quantizer import MMDVectorQuantizer
from utils.util import Pack
from models.lpips import LPIPS

def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class VQModel(nn.Module):
    def __init__(self, args):
        super(VQModel, self).__init__()
        self.args = args
        self.projector_in = nn.Sequential(
                nn.Conv2d(48, 1024, kernel_size=3, padding=1),
                Normalize(1024),
                nn.SiLU(),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                Normalize(1024),
                nn.SiLU(),
                nn.Conv2d(1024, args.codebook_dim, kernel_size=3, padding=1),
            )
        self.projector_out = nn.Sequential(
                nn.Conv2d(args.codebook_dim, 1024, kernel_size=3, padding=1),
                Normalize(1024),
                nn.SiLU(),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                Normalize(1024),
                nn.SiLU(),
                nn.Conv2d(1024, 48, kernel_size=3, padding=1),
            ) 

        if args.VQ == "vanilla_vq":
            self.quantizer = VanillaVectorQuantizer(args)
        elif args.VQ == "ema_vq":
            self.quantizer = EMAVectorQuantizer(args)
        elif args.VQ == "online_vq":
            self.quantizer = OnlineVectorQuantizer(args)
        elif args.VQ == "wasserstein_vq":
            self.quantizer = WassersteinVectorQuantizer(args)
        elif args.VQ == "mmd_vq":
            self.quantizer = MMDVectorQuantizer(args)
        self.perceptual_loss = LPIPS().eval()
        self.downscale = nn.PixelUnshuffle(downscale_factor=4)
        self.upscale = nn.PixelShuffle(upscale_factor=4)

    def forward(self, x):
        x_down =  self.downscale(x)          ## [B, C, H, W] to [B, C*r*r, H/r, W/r]
        ze = self.projector_in(x_down)       ## [B, C*r*r, H/r, W/r] to [B, project_dim, H/r, W/r]

        zq, vq_loss, utilization, perplexity = self.quantizer(ze)
        zq = self.projector_out(zq)  ##  [B, project_dim, H/r, W/r] to [B, C*r*r, H/r, W/r] 
        x_rec = self.upscale(zq)     ##  [B, C*r*r, H/r, W/r] to [B, C, H, W]

        p_loss = self.perceptual_loss(x.contiguous(), x_rec.contiguous())
        p_loss = torch.mean(p_loss)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())

        loss = rec_loss + p_loss + vq_loss
        return loss, rec_loss, p_loss, utilization, perplexity

    def collect_eval_info(self, x):
        x_down =  self.downscale(x)          ## [B, C, H, W] to [B, C*r*r, H/r, W/r]
        ze = self.projector_in(x_down)       ## [B, C*r*r, H/r, W/r] to [B, project_dim, H/r, W/r]

        zq, histogram = self.quantizer.collect_eval_info(ze)
        zq = self.projector_out(zq)                ##  [B, project_dim, H/r, W/r] to [B, C*r*r, H/r, W/r] 
        x_rec = self.upscale(zq).clamp_(-1, 1)     ##  [B, C*r*r, H/r, W/r] to [B, C, H, W]

        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        return x_rec, rec_loss, histogram

    def reconstruction(self, x):
        x_down =  self.downscale(x)          ## [B, C, H, W] to [B, C*r*r, H/r, W/r]
        ze = self.projector_in(x_down)       ## [B, C*r*r, H/r, W/r] to [B, project_dim, H/r, W/r]

        zq = self.quantizer.collect_reconstruction(ze)
        zq = self.projector_out(zq)                ##  [B, project_dim, H/r, W/r] to [B, C*r*r, H/r, W/r] 
        x_rec = self.upscale(zq).clamp_(-1, 1)     ##  [B, C*r*r, H/r, W/r] to [B, C, H, W]
        return x_rec

