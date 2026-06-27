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

    if args.stage == "transplant":
        ssim, psnr, lpips, rec_loss, quant_error, utilization, perplexity, total_num =  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        if args.chunks == 1:
            histogram_all_1: torch.Tensor = 0.0
        elif args.chunks == 2:
            histogram_all_1: torch.Tensor = 0.0
            histogram_all_2: torch.Tensor = 0.0
        elif args.chunks == 4:
            histogram_all_1: torch.Tensor = 0.0
            histogram_all_2: torch.Tensor = 0.0
            histogram_all_3: torch.Tensor = 0.0
            histogram_all_4: torch.Tensor = 0.0
        elif args.chunks == 8:
            histogram_all_1: torch.Tensor = 0.0
            histogram_all_2: torch.Tensor = 0.0
            histogram_all_3: torch.Tensor = 0.0
            histogram_all_4: torch.Tensor = 0.0
            histogram_all_5: torch.Tensor = 0.0
            histogram_all_6: torch.Tensor = 0.0
            histogram_all_7: torch.Tensor = 0.0
            histogram_all_8: torch.Tensor = 0.0
    if args.stage == "refinement":
        ssim, psnr, lpips, rec_loss, total_num = 0.0, 0.0, 0.0, 0.0, 0 
    
    for step, (x, _) in enumerate(val_dataloader):
        x = x.cuda(int(os.environ['LOCAL_RANK']), non_blocking=True)
        batch_size = x.size(0)
        total_num += batch_size
        with torch.no_grad():
            if args.stage == "transplant":
                if args.chunks == 1:
                    x_rec, rec_loss_eval, quant_error_eval, histogram_eval_1 = model.module.collect_eval_info_transplant(x)
                    histogram_all_1 += histogram_eval_1
                elif args.chunks == 2:
                    x_rec, rec_loss_eval, quant_error_eval, histogram_eval_1, histogram_eval_2 = model.module.collect_eval_info_transplant(x)
                    histogram_all_1 += histogram_eval_1
                    histogram_all_2 += histogram_eval_2
                elif args.chunks == 4:
                    x_rec, rec_loss_eval, quant_error_eval, histogram_eval_1, histogram_eval_2, histogram_eval_3, histogram_eval_4 = model.module.collect_eval_info_transplant(x)
                    histogram_all_1 += histogram_eval_1
                    histogram_all_2 += histogram_eval_2
                    histogram_all_3 += histogram_eval_3
                    histogram_all_4 += histogram_eval_4
                elif args.chunks == 8:
                    x_rec, rec_loss_eval, quant_error_eval, histogram_eval_1, histogram_eval_2, histogram_eval_3, histogram_eval_4, histogram_eval_5, histogram_eval_6, histogram_eval_7, histogram_eval_8 = model.module.collect_eval_info_transplant(x)
                    histogram_all_1 += histogram_eval_1
                    histogram_all_2 += histogram_eval_2
                    histogram_all_3 += histogram_eval_3
                    histogram_all_4 += histogram_eval_4
                    histogram_all_5 += histogram_eval_5
                    histogram_all_6 += histogram_eval_6
                    histogram_all_7 += histogram_eval_7
                    histogram_all_8 += histogram_eval_8
                info_pack = Pack(rec_loss=rec_loss_eval, quant_error=quant_error_eval)
            else:
                x_rec, rec_loss_eval = model.module.collect_eval_info_refinement(x)
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
            if args.stage == "transplant":
                quant_error += info_pack.quant_error.item() * batch_size
                
    eval_psnr = psnr/len_val_set
    eval_ssim = ssim/len_val_set
    eval_lpips = lpips/len_val_set
    eval_rec_loss = rec_loss/total_num
    if args.stage == "transplant":
        eval_quant_error = quant_error/total_num
        if args.chunks == 1:
            codebook_usage_counts_1 = (histogram_all_1 > 0).float().sum()
            eval_utilization  = codebook_usage_counts_1.item() / args.codebook_size
            avg_probs_1 = histogram_all_1/histogram_all_1.sum(0)
            eval_perplexity = torch.exp(-torch.sum(avg_probs_1 * torch.log(avg_probs_1 + 1e-10))).item()

        elif args.chunks == 2: 
            codebook_usage_counts_1 = (histogram_all_1 > 0).float().sum()
            eval_utilization_1  = codebook_usage_counts_1.item() / args.codebook_size
            avg_probs_1 = histogram_all_1/histogram_all_1.sum(0)
            eval_perplexity_1 = torch.exp(-torch.sum(avg_probs_1 * torch.log(avg_probs_1 + 1e-10))).item()

            codebook_usage_counts_2 = (histogram_all_2 > 0).float().sum()
            eval_utilization_2  = codebook_usage_counts_2.item() / args.codebook_size
            avg_probs_2 = histogram_all_2/histogram_all_2.sum(0)
            eval_perplexity_2 = torch.exp(-torch.sum(avg_probs_2 * torch.log(avg_probs_2 + 1e-10))).item()

            eval_utilization = 0.5 * (eval_utilization_1 + eval_utilization_2)
            eval_perplexity = 0.5 * (eval_perplexity_1 + eval_perplexity_2)
        
        elif args.chunks == 4: 
            codebook_usage_counts_1 = (histogram_all_1 > 0).float().sum()
            eval_utilization_1  = codebook_usage_counts_1.item() / args.codebook_size
            avg_probs_1 = histogram_all_1/histogram_all_1.sum(0)
            eval_perplexity_1 = torch.exp(-torch.sum(avg_probs_1 * torch.log(avg_probs_1 + 1e-10))).item()

            codebook_usage_counts_2 = (histogram_all_2 > 0).float().sum()
            eval_utilization_2  = codebook_usage_counts_2.item() / args.codebook_size
            avg_probs_2 = histogram_all_2/histogram_all_2.sum(0)
            eval_perplexity_2 = torch.exp(-torch.sum(avg_probs_2 * torch.log(avg_probs_2 + 1e-10))).item()

            codebook_usage_counts_3 = (histogram_all_3 > 0).float().sum()
            eval_utilization_3  = codebook_usage_counts_3.item() / args.codebook_size
            avg_probs_3 = histogram_all_3/histogram_all_3.sum(0)
            eval_perplexity_3 = torch.exp(-torch.sum(avg_probs_3 * torch.log(avg_probs_3 + 1e-10))).item()

            codebook_usage_counts_4 = (histogram_all_4 > 0).float().sum()
            eval_utilization_4  = codebook_usage_counts_4.item() / args.codebook_size
            avg_probs_4 = histogram_all_4/histogram_all_4.sum(0)
            eval_perplexity_4 = torch.exp(-torch.sum(avg_probs_4 * torch.log(avg_probs_4 + 1e-10))).item()

            eval_utilization = 0.25 * (eval_utilization_1 + eval_utilization_2 + eval_utilization_3 + eval_utilization_4)
            eval_perplexity = 0.25 * (eval_perplexity_1 + eval_perplexity_2 + eval_perplexity_3 + eval_perplexity_4)

        elif args.chunks == 8: 
            codebook_usage_counts_1 = (histogram_all_1 > 0).float().sum()
            eval_utilization_1  = codebook_usage_counts_1.item() / args.codebook_size
            avg_probs_1 = histogram_all_1/histogram_all_1.sum(0)
            eval_perplexity_1 = torch.exp(-torch.sum(avg_probs_1 * torch.log(avg_probs_1 + 1e-10))).item()

            codebook_usage_counts_2 = (histogram_all_2 > 0).float().sum()
            eval_utilization_2  = codebook_usage_counts_2.item() / args.codebook_size
            avg_probs_2 = histogram_all_2/histogram_all_2.sum(0)
            eval_perplexity_2 = torch.exp(-torch.sum(avg_probs_2 * torch.log(avg_probs_2 + 1e-10))).item()

            codebook_usage_counts_3 = (histogram_all_3 > 0).float().sum()
            eval_utilization_3  = codebook_usage_counts_3.item() / args.codebook_size
            avg_probs_3 = histogram_all_3/histogram_all_3.sum(0)
            eval_perplexity_3 = torch.exp(-torch.sum(avg_probs_3 * torch.log(avg_probs_3 + 1e-10))).item()

            codebook_usage_counts_4 = (histogram_all_4 > 0).float().sum()
            eval_utilization_4  = codebook_usage_counts_4.item() / args.codebook_size
            avg_probs_4 = histogram_all_4/histogram_all_4.sum(0)
            eval_perplexity_4 = torch.exp(-torch.sum(avg_probs_4 * torch.log(avg_probs_4 + 1e-10))).item()

            codebook_usage_counts_5 = (histogram_all_5 > 0).float().sum()
            eval_utilization_5  = codebook_usage_counts_5.item() / args.codebook_size
            avg_probs_5 = histogram_all_5/histogram_all_5.sum(0)
            eval_perplexity_5 = torch.exp(-torch.sum(avg_probs_5 * torch.log(avg_probs_5 + 1e-10))).item()

            codebook_usage_counts_6 = (histogram_all_6 > 0).float().sum()
            eval_utilization_6  = codebook_usage_counts_6.item() / args.codebook_size
            avg_probs_6 = histogram_all_6/histogram_all_6.sum(0)
            eval_perplexity_6 = torch.exp(-torch.sum(avg_probs_6 * torch.log(avg_probs_6 + 1e-10))).item()

            codebook_usage_counts_7 = (histogram_all_7 > 0).float().sum()
            eval_utilization_7  = codebook_usage_counts_7.item() / args.codebook_size
            avg_probs_7 = histogram_all_7/histogram_all_7.sum(0)
            eval_perplexity_7 = torch.exp(-torch.sum(avg_probs_7 * torch.log(avg_probs_7 + 1e-10))).item()

            codebook_usage_counts_8 = (histogram_all_8 > 0).float().sum()
            eval_utilization_8  = codebook_usage_counts_8.item() / args.codebook_size
            avg_probs_8 = histogram_all_8/histogram_all_8.sum(0)
            eval_perplexity_8 = torch.exp(-torch.sum(avg_probs_8 * torch.log(avg_probs_8 + 1e-10))).item()

            eval_utilization = 0.125 * (eval_utilization_1 + eval_utilization_2 + eval_utilization_3 + eval_utilization_4 + eval_utilization_5 + eval_utilization_6 + eval_utilization_7 + eval_utilization_8)
            eval_perplexity = 0.125 * (eval_perplexity_1 + eval_perplexity_2 + eval_perplexity_3 + eval_perplexity_4 + eval_perplexity_5 + eval_perplexity_6 + eval_perplexity_7 + eval_perplexity_8)

    model.train()
    if args.stage == "transplant":
        model.module.encoder.eval()
        model.module.decoder.eval()
        model.module.quant_conv.eval()
        model.module.post_quant_conv.eval()
        model.module.perceptual_loss.eval()
    else:
        model.module.encoder.eval()
        model.module.quant_conv.eval()
        model.module.projector_in.eval()
        if args.chunks == 1:
            model.module.quantizer1.eval()
        elif args.chunks == 2: 
            model.module.quantizer1.eval()  
            model.module.quantizer2.eval()
        elif args.chunks == 4: 
            model.module.quantizer1.eval()  
            model.module.quantizer2.eval()
            model.module.quantizer3.eval()  
            model.module.quantizer4.eval()
        elif args.chunks == 8: 
            model.module.quantizer1.eval()  
            model.module.quantizer2.eval()
            model.module.quantizer3.eval()  
            model.module.quantizer4.eval()
            model.module.quantizer5.eval()  
            model.module.quantizer6.eval()
            model.module.quantizer7.eval()  
            model.module.quantizer8.eval()
        
    if args.stage == "transplant":
        return Pack(psnr=eval_psnr, ssim=eval_ssim, lpips=eval_lpips, rec_loss=eval_rec_loss, quant_error=eval_quant_error, utilization=eval_utilization, perplexity=eval_perplexity)
    else:
        return Pack(psnr=eval_psnr, ssim=eval_ssim, lpips=eval_lpips, rec_loss=eval_rec_loss)

## product_quantizer
def eval_one_epoch_pq(args, model, epoch, val_dataloader, len_val_set):
    model.eval()
    psnr_metric = PSNR()
    ssim_metric = SSIM()
    lpips_metric = LPIPS()
    total_codebook_size = args.codebook_size * args.codebook_size

    if args.stage == "transplant":
        ssim, psnr, lpips, rec_loss, quant_error, utilization, perplexity, total_num =  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        histogram_all_1: torch.Tensor = 0.0
        histogram_all_2: torch.Tensor = 0.0
    if args.stage == "refinement":
        ssim, psnr, lpips, rec_loss, total_num = 0.0, 0.0, 0.0, 0.0, 0 
    
    for step, (x, _) in enumerate(val_dataloader):
        x = x.cuda(int(os.environ['LOCAL_RANK']), non_blocking=True)
        batch_size = x.size(0)
        total_num += batch_size
        with torch.no_grad():
            if args.stage == "transplant":
                x_rec, rec_loss_eval, quant_error_eval, histogram_eval_1, histogram_eval_2 = model.module.collect_eval_info_transplant(x)
                info_pack = Pack(rec_loss=rec_loss_eval, quant_error=quant_error_eval)
                histogram_all_1 += histogram_eval_1
                histogram_all_2 += histogram_eval_2
            else:
                x_rec, rec_loss_eval = model.module.collect_eval_info_refinement(x)
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
            if args.stage == "transplant":
                quant_error += info_pack.quant_error.item() * batch_size
                
    eval_psnr = psnr/len_val_set
    eval_ssim = ssim/len_val_set
    eval_lpips = lpips/len_val_set
    eval_rec_loss = rec_loss/total_num
    if args.stage == "transplant":
        eval_quant_error = quant_error/total_num
        codebook_usage_counts_1 = (histogram_all_1 > 0).float().sum()
        eval_utilization_1  = codebook_usage_counts_1.item() / total_codebook_size
        avg_probs_1 = histogram_all_1/histogram_all_1.sum(0)
        eval_perplexity_1 = torch.exp(-torch.sum(avg_probs_1 * torch.log(avg_probs_1 + 1e-10))).item()

        codebook_usage_counts_2 = (histogram_all_2 > 0).float().sum()
        eval_utilization_2  = codebook_usage_counts_2.item() / total_codebook_size
        avg_probs_2 = histogram_all_2/histogram_all_2.sum(0)
        eval_perplexity_2 = torch.exp(-torch.sum(avg_probs_2 * torch.log(avg_probs_2 + 1e-10))).item()

        eval_utilization = 0.5 * (eval_utilization_1 + eval_utilization_2)
        eval_perplexity = 0.5 * (eval_perplexity_1 + eval_perplexity_2)

    model.train()
    if args.stage == "transplant":
        model.module.encoder.eval()
        model.module.decoder.eval()
        model.module.quant_conv.eval()
        model.module.post_quant_conv.eval()
        model.module.perceptual_loss.eval()
    else:
        model.module.encoder.eval()
        model.module.quant_conv.eval()
        model.module.quantizer1.eval()
        model.module.quantizer2.eval()
        model.module.projector_in.eval()
        
    if args.stage == "transplant":
        return Pack(psnr=eval_psnr, ssim=eval_ssim, lpips=eval_lpips, rec_loss=eval_rec_loss, quant_error=eval_quant_error, utilization=eval_utilization, perplexity=eval_perplexity)
    else:
        return Pack(psnr=eval_psnr, ssim=eval_ssim, lpips=eval_lpips, rec_loss=eval_rec_loss)

## scalar_quantizer
def eval_one_epoch_sq(args, model, epoch, val_dataloader, len_val_set):
    model.eval()
    psnr_metric = PSNR()
    ssim_metric = SSIM()
    lpips_metric = LPIPS()
    total_codebook_size = args.L ** args.project_dim

    if args.stage == "transplant":
        ssim, psnr, lpips, rec_loss, quant_error, utilization, perplexity, total_num =  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        histogram_all_1: torch.Tensor = 0.0
        histogram_all_2: torch.Tensor = 0.0
    if args.stage == "refinement":
        ssim, psnr, lpips, rec_loss, total_num = 0.0, 0.0, 0.0, 0.0, 0 
    
    for step, (x, _) in enumerate(val_dataloader):
        x = x.cuda(int(os.environ['LOCAL_RANK']), non_blocking=True)
        batch_size = x.size(0)
        total_num += batch_size
        with torch.no_grad():
            if args.stage == "transplant":
                x_rec, rec_loss_eval, quant_error_eval, histogram_eval_1, histogram_eval_2 = model.module.collect_eval_info_transplant(x)
                info_pack = Pack(rec_loss=rec_loss_eval, quant_error=quant_error_eval)
                histogram_all_1 += histogram_eval_1
                histogram_all_2 += histogram_eval_2
            else:
                x_rec, rec_loss_eval = model.module.collect_eval_info_refinement(x)
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
            if args.stage == "transplant":
                quant_error += info_pack.quant_error.item() * batch_size
                
    eval_psnr = psnr/len_val_set
    eval_ssim = ssim/len_val_set
    eval_lpips = lpips/len_val_set
    eval_rec_loss = rec_loss/total_num
    if args.stage == "transplant":
        eval_quant_error = quant_error/total_num
        codebook_usage_counts_1 = (histogram_all_1 > 0).float().sum()
        eval_utilization_1  = codebook_usage_counts_1.item() / total_codebook_size
        avg_probs_1 = histogram_all_1/histogram_all_1.sum(0)
        eval_perplexity_1 = torch.exp(-torch.sum(avg_probs_1 * torch.log(avg_probs_1 + 1e-10))).item()

        codebook_usage_counts_2 = (histogram_all_2 > 0).float().sum()
        eval_utilization_2  = codebook_usage_counts_2.item() / total_codebook_size
        avg_probs_2 = histogram_all_2/histogram_all_2.sum(0)
        eval_perplexity_2 = torch.exp(-torch.sum(avg_probs_2 * torch.log(avg_probs_2 + 1e-10))).item()

        eval_utilization = 0.5 * (eval_utilization_1 + eval_utilization_2)
        eval_perplexity = 0.5 * (eval_perplexity_1 + eval_perplexity_2)

    model.train()
    if args.stage == "transplant":
        model.module.encoder.eval()
        model.module.decoder.eval()
        model.module.quant_conv.eval()
        model.module.post_quant_conv.eval()
        model.module.perceptual_loss.eval()
    else:
        model.module.encoder.eval()
        model.module.quant_conv.eval()
        model.module.projector_in.eval()
        
    if args.stage == "transplant":
        return Pack(psnr=eval_psnr, ssim=eval_ssim, lpips=eval_lpips, rec_loss=eval_rec_loss, quant_error=eval_quant_error, utilization=eval_utilization, perplexity=eval_perplexity)
    else:
        return Pack(psnr=eval_psnr, ssim=eval_ssim, lpips=eval_lpips, rec_loss=eval_rec_loss)