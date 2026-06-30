import gc
import os
import shutil
import sys
import time
import warnings
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch import nn, optim
import math
import json
import random
import scipy.io as sio
from torch.nn import functional as F
from scipy.io import savemat
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from data.dataloader import build_dataloader
import torchvision.models as torchvision_models
from torchvision import models, datasets, transforms
from torch import distributed as tdist
import itertools
from copy import deepcopy

import config
from utils.util import Logger, LossManager, Pack, adjust_learning_rate
from data import dataloader
from metric.metric import PSNR, LPIPS, SSIM
import warnings
warnings.filterwarnings('ignore')

## for vector quantizer
def eval_one_epoch_vq(args, model, epoch, val_dataloader, len_val_set):
    model.eval()
    psnr_metric = PSNR()
    ssim_metric = SSIM()
    lpips_metric = LPIPS()

    ssim, psnr, lpips, rec_loss, utilization, perplexity, total_num =  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    histogram_all: torch.Tensor = 0.0

    for step, (x, _) in enumerate(val_dataloader):
        x = x.cuda(int(os.environ['LOCAL_RANK']), non_blocking=True)
        batch_size = x.size(0)
        total_num += batch_size
        with torch.no_grad():
            x_rec, rec_loss_eval, histogram_eval = model.module.collect_eval_info(x)
            histogram_all += histogram_eval
            info_pack = Pack(rec_loss=rec_loss_eval)

            x_norm = (x + 1.0)/2.0
            x_rec_norm = (x_rec + 1.0)/2.0
            batch_lpips = lpips_metric(x_norm, x_rec_norm).sum()
            batch_psnr = psnr_metric(x_norm, x_rec_norm).sum()
            batch_ssim = ssim_metric(x_norm, x_rec_norm).sum()

        handler1 = tdist.all_reduce(batch_lpips, async_op=True)
        handler2 = tdist.all_reduce(batch_psnr, async_op=True)
        handler3 = tdist.all_reduce(batch_ssim, async_op=True)
        handler1.wait()
        handler2.wait()
        handler3.wait()

        if int(os.environ['LOCAL_RANK']) == 0:
            ssim += batch_ssim.item()
            psnr += batch_psnr.item()
            lpips += batch_lpips.item()
            rec_loss += info_pack.rec_loss.item() * batch_size
                
    eval_psnr = psnr/len_val_set
    eval_ssim = ssim/len_val_set
    eval_lpips = lpips/len_val_set
    eval_rec_loss = rec_loss/total_num
    codebook_usage_counts = (histogram_all > 0).float().sum()
    eval_utilization  = codebook_usage_counts.item() / args.codebook_size
    avg_probs = histogram_all/histogram_all.sum(0)
    eval_perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))).item()
        
    model.train()
    model.module.perceptual_loss.eval()
    return Pack(psnr=eval_psnr, ssim=eval_ssim, lpips=eval_lpips, rec_loss=eval_rec_loss, utilization=eval_utilization, perplexity=eval_perplexity)


## product_quantizer
def eval_one_epoch_pq(args, model, epoch, val_dataloader, len_val_set):
    model.eval()
    psnr_metric = PSNR()
    ssim_metric = SSIM()
    lpips_metric = LPIPS()
    total_codebook_size = args.codebook_size * args.codebook_size

    ssim, psnr, lpips, rec_loss, utilization, perplexity, total_num =  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    histogram_all: torch.Tensor = 0.0
    for step, (x, _) in enumerate(val_dataloader):
        x = x.cuda(int(os.environ['LOCAL_RANK']), non_blocking=True)
        batch_size = x.size(0)
        total_num += batch_size
        with torch.no_grad():
            x_rec, rec_loss_eval, histogram_eval = model.module.collect_eval_info(x)
            histogram_all += histogram_eval
            info_pack = Pack(rec_loss=rec_loss_eval)

            x_norm = (x + 1.0)/2.0
            x_rec_norm = (x_rec + 1.0)/2.0
            batch_lpips = lpips_metric(x_norm, x_rec_norm).sum()
            batch_psnr = psnr_metric(x_norm, x_rec_norm).sum()
            batch_ssim = ssim_metric(x_norm, x_rec_norm).sum()

        handler1 = tdist.all_reduce(batch_lpips, async_op=True)
        handler2 = tdist.all_reduce(batch_psnr, async_op=True)
        handler3 = tdist.all_reduce(batch_ssim, async_op=True)
        handler1.wait()
        handler2.wait()
        handler3.wait()

        if int(os.environ['LOCAL_RANK']) == 0:
            ssim += batch_ssim.item()
            psnr += batch_psnr.item()
            lpips += batch_lpips.item()
            rec_loss += info_pack.rec_loss.item() * batch_size
                
    eval_psnr = psnr/len_val_set
    eval_ssim = ssim/len_val_set
    eval_lpips = lpips/len_val_set
    eval_rec_loss = rec_loss/total_num
    codebook_usage_counts = (histogram_all > 0).float().sum()
    eval_utilization  = codebook_usage_counts.item() / total_codebook_size
    avg_probs = histogram_all/histogram_all.sum(0)
    eval_perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))).item()

    model.train()
    model.module.perceptual_loss.eval()
    return Pack(psnr=eval_psnr, ssim=eval_ssim, lpips=eval_lpips, rec_loss=eval_rec_loss, utilization=eval_utilization, perplexity=eval_perplexity)

## scalar_quantizer
def eval_one_epoch_sq(args, model, epoch, val_dataloader, len_val_set):
    model.eval()
    psnr_metric = PSNR()
    ssim_metric = SSIM()
    lpips_metric = LPIPS()
    total_codebook_size = args.L ** args.project_dim

    ssim, psnr, lpips, rec_loss, utilization, perplexity, total_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    histogram_all: torch.Tensor = 0.0
    for step, (x, _) in enumerate(val_dataloader):
        x = x.cuda(int(os.environ['LOCAL_RANK']), non_blocking=True)
        batch_size = x.size(0)
        total_num += batch_size
        with torch.no_grad():
            x_rec, rec_loss_eval, histogram_eval = model.module.collect_eval_info(x)
            histogram_all += histogram_eval
            info_pack = Pack(rec_loss=rec_loss_eval)

            x_norm = (x + 1.0)/2.0
            x_rec_norm = (x_rec + 1.0)/2.0
            batch_lpips = lpips_metric(x_norm, x_rec_norm).sum()
            batch_psnr = psnr_metric(x_norm, x_rec_norm).sum()
            batch_ssim = ssim_metric(x_norm, x_rec_norm).sum()

        handler1 = tdist.all_reduce(batch_lpips, async_op=True)
        handler2 = tdist.all_reduce(batch_psnr, async_op=True)
        handler3 = tdist.all_reduce(batch_ssim, async_op=True)
        handler1.wait()
        handler2.wait()
        handler3.wait()

        if int(os.environ['LOCAL_RANK']) == 0:
            ssim += batch_ssim.item()
            psnr += batch_psnr.item()
            lpips += batch_lpips.item()
            rec_loss += info_pack.rec_loss.item() * batch_size
                
    eval_psnr = psnr/len_val_set
    eval_ssim = ssim/len_val_set
    eval_lpips = lpips/len_val_set
    eval_rec_loss = rec_loss/total_num
    codebook_usage_counts = (histogram_all > 0).float().sum()
    eval_utilization  = codebook_usage_counts.item() / total_codebook_size
    avg_probs = histogram_all/histogram_all.sum(0)
    eval_perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))).item()

    model.train()
    model.module.perceptual_loss.eval()
    return Pack(psnr=eval_psnr, ssim=eval_ssim, lpips=eval_lpips, rec_loss=eval_rec_loss, utilization=eval_utilization, perplexity=eval_perplexity)
    