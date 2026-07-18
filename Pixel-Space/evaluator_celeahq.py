import os
import torch
import warnings
import random
import numpy as np
import PIL.Image as PImage
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image, ImageOps, ImageFilter

import config
from cleanfid import fid

input_dir = "/project/6105494/shared/reconstruction/CelebAHQ"

print("#################VQ method###########################")
## Vanilla VQ
Vanilla_VQ_5 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/vanilla_vq_65536_5"
Vanilla_VQ_10 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/vanilla_vq_65536_10"
Vanilla_VQ_15 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/vanilla_vq_65536_15"
Vanilla_VQ_20 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/vanilla_vq_65536_20"
Vanilla_VQ_25 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/vanilla_vq_65536_25"
Vanilla_VQ_30 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/vanilla_vq_65536_30"
print(Vanilla_VQ_5)
FID = fid.compute_fid(Vanilla_VQ_5, input_dir)
print("FID: "+str(FID))

print(Vanilla_VQ_10)
FID = fid.compute_fid(Vanilla_VQ_10, input_dir)
print("FID: "+str(FID))

print(Vanilla_VQ_15)
FID = fid.compute_fid(Vanilla_VQ_15, input_dir)
print("FID: "+str(FID))

print(Vanilla_VQ_20)
FID = fid.compute_fid(Vanilla_VQ_20, input_dir)
print("FID: "+str(FID))

print(Vanilla_VQ_25)
FID = fid.compute_fid(Vanilla_VQ_25, input_dir)
print("FID: "+str(FID))

print(Vanilla_VQ_30)
FID = fid.compute_fid(Vanilla_VQ_30, input_dir)
print("FID: "+str(FID))

##EMA VQ
EMA_VQ_5 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/ema_vq_65536_5"
EMA_VQ_10 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/ema_vq_65536_10"
EMA_VQ_15 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/ema_vq_65536_15"
EMA_VQ_20 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/ema_vq_65536_20"
EMA_VQ_25 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/ema_vq_65536_25"
EMA_VQ_30 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/ema_vq_65536_30"
print(EMA_VQ_5)
FID = fid.compute_fid(EMA_VQ_5, input_dir)
print("FID: "+str(FID))

print(EMA_VQ_10)
FID = fid.compute_fid(EMA_VQ_10, input_dir)
print("FID: "+str(FID))

print(EMA_VQ_15)
FID = fid.compute_fid(EMA_VQ_15, input_dir)
print("FID: "+str(FID))

print(EMA_VQ_20)
FID = fid.compute_fid(EMA_VQ_20, input_dir)
print("FID: "+str(FID))

print(EMA_VQ_25)
FID = fid.compute_fid(EMA_VQ_25, input_dir)
print("FID: "+str(FID))

print(EMA_VQ_30)
FID = fid.compute_fid(EMA_VQ_30, input_dir)
print("FID: "+str(FID))

##Wasserstein VQ
Wasserstein_VQ_5 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/wasserstein_vq_65536_5"
Wasserstein_VQ_10 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/wasserstein_vq_65536_10"
Wasserstein_VQ_15 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/wasserstein_vq_65536_15"
Wasserstein_VQ_20 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/wasserstein_vq_65536_20"
Wasserstein_VQ_25 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/wasserstein_vq_65536_25"
Wasserstein_VQ_30 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/wasserstein_vq_65536_30"
print(Wasserstein_VQ_5)
FID = fid.compute_fid(Wasserstein_VQ_5, input_dir)
print("FID: "+str(FID))

print(Wasserstein_VQ_10)
FID = fid.compute_fid(Wasserstein_VQ_10, input_dir)
print("FID: "+str(FID))

print(Wasserstein_VQ_15)
FID = fid.compute_fid(Wasserstein_VQ_15, input_dir)
print("FID: "+str(FID))

print(Wasserstein_VQ_20)
FID = fid.compute_fid(Wasserstein_VQ_20, input_dir)
print("FID: "+str(FID))

print(Wasserstein_VQ_25)
FID = fid.compute_fid(Wasserstein_VQ_25, input_dir)
print("FID: "+str(FID))

print(Wasserstein_VQ_30)
FID = fid.compute_fid(Wasserstein_VQ_30, input_dir)
print("FID: "+str(FID))

##MMD VQ
MMD_VQ_5 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/mmd_vq_65536_5"
MMD_VQ_10 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/mmd_vq_65536_10"
MMD_VQ_15 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/mmd_vq_65536_15"
MMD_VQ_20 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/mmd_vq_65536_20"
MMD_VQ_25 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/mmd_vq_65536_25"
MMD_VQ_30 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/mmd_vq_65536_30"
print(MMD_VQ_5)
FID = fid.compute_fid(MMD_VQ_5, input_dir)
print("FID: "+str(FID))

print(MMD_VQ_10)
FID = fid.compute_fid(MMD_VQ_10, input_dir)
print("FID: "+str(FID))

print(MMD_VQ_15)
FID = fid.compute_fid(MMD_VQ_15, input_dir)
print("FID: "+str(FID))

print(MMD_VQ_20)
FID = fid.compute_fid(MMD_VQ_20, input_dir)
print("FID: "+str(FID))

print(MMD_VQ_25)
FID = fid.compute_fid(MMD_VQ_25, input_dir)
print("FID: "+str(FID))

print(MMD_VQ_30)
FID = fid.compute_fid(MMD_VQ_30, input_dir)
print("FID: "+str(FID))



print("#################PQ method###########################")
##Vanilla PQ
Vanilla_PQ_5 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/vanilla_vq_2_5"
Vanilla_PQ_10 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/vanilla_vq_2_10"
Vanilla_PQ_15 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/vanilla_vq_2_15"
Vanilla_PQ_20 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/vanilla_vq_2_20"
Vanilla_PQ_25 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/vanilla_vq_2_25"
Vanilla_PQ_30 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/vanilla_vq_2_30"
print(Vanilla_PQ_5)
FID = fid.compute_fid(Vanilla_PQ_5, input_dir)
print("FID: "+str(FID))

print(Vanilla_PQ_10)
FID = fid.compute_fid(Vanilla_PQ_10, input_dir)
print("FID: "+str(FID))

print(Vanilla_PQ_15)
FID = fid.compute_fid(Vanilla_PQ_15, input_dir)
print("FID: "+str(FID))

print(Vanilla_PQ_20)
FID = fid.compute_fid(Vanilla_PQ_20, input_dir)
print("FID: "+str(FID))

print(Vanilla_PQ_25)
FID = fid.compute_fid(Vanilla_PQ_25, input_dir)
print("FID: "+str(FID))

print(Vanilla_PQ_30)
FID = fid.compute_fid(Vanilla_PQ_30, input_dir)
print("FID: "+str(FID))


##EMA PQ
EMA_PQ_5 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/ema_vq_2_5"
EMA_PQ_10 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/ema_vq_2_10"
EMA_PQ_15 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/ema_vq_2_15"
EMA_PQ_20 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/ema_vq_2_20"
EMA_PQ_25 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/ema_vq_2_25"
EMA_PQ_30 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/ema_vq_2_30"
print(EMA_PQ_5)
FID = fid.compute_fid(EMA_PQ_5, input_dir)
print("FID: "+str(FID))

print(EMA_PQ_10)
FID = fid.compute_fid(EMA_PQ_10, input_dir)
print("FID: "+str(FID))

print(EMA_PQ_15)
FID = fid.compute_fid(EMA_PQ_15, input_dir)
print("FID: "+str(FID))

print(EMA_PQ_20)
FID = fid.compute_fid(EMA_PQ_20, input_dir)
print("FID: "+str(FID))

print(EMA_PQ_25)
FID = fid.compute_fid(EMA_PQ_25, input_dir)
print("FID: "+str(FID))

print(EMA_PQ_30)
FID = fid.compute_fid(EMA_PQ_30, input_dir)
print("FID: "+str(FID))

##Online PQ
Online_PQ_5 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/online_vq_2_5"
Online_PQ_10 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/online_vq_2_10"
Online_PQ_15 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/online_vq_2_15"
Online_PQ_20 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/online_vq_2_20"
Online_PQ_25 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/online_vq_2_25"
Online_PQ_30 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/online_vq_2_30"

print(Online_PQ_5)
FID = fid.compute_fid(Online_PQ_5, input_dir)
print("FID: "+str(FID))

print(Online_PQ_10)
FID = fid.compute_fid(Online_PQ_10, input_dir)
print("FID: "+str(FID))

print(Online_PQ_15)
FID = fid.compute_fid(Online_PQ_15, input_dir)
print("FID: "+str(FID))

print(Online_PQ_20)
FID = fid.compute_fid(Online_PQ_20, input_dir)
print("FID: "+str(FID))

print(Online_PQ_25)
FID = fid.compute_fid(Online_PQ_25, input_dir)
print("FID: "+str(FID))

print(Online_PQ_30)
FID = fid.compute_fid(Online_PQ_30, input_dir)
print("FID: "+str(FID))

##Wasserstein PQ
Wasserstein_PQ_5 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/wasserstein_vq_2_5"
Wasserstein_PQ_10 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/wasserstein_vq_2_10"
Wasserstein_PQ_15 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/wasserstein_vq_2_15"
Wasserstein_PQ_20 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/wasserstein_vq_2_20"
Wasserstein_PQ_25 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/wasserstein_vq_2_25"
Wasserstein_PQ_30 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/wasserstein_vq_2_30"
print(Wasserstein_PQ_5)
FID = fid.compute_fid(Wasserstein_PQ_5, input_dir)
print("FID: "+str(FID))

print(Wasserstein_PQ_10)
FID = fid.compute_fid(Wasserstein_PQ_10, input_dir)
print("FID: "+str(FID))

print(Wasserstein_PQ_15)
FID = fid.compute_fid(Wasserstein_PQ_15, input_dir)
print("FID: "+str(FID))

print(Wasserstein_PQ_20)
FID = fid.compute_fid(Wasserstein_PQ_20, input_dir)
print("FID: "+str(FID))

print(Wasserstein_PQ_25)
FID = fid.compute_fid(Wasserstein_PQ_25, input_dir)
print("FID: "+str(FID))

print(Wasserstein_PQ_30)
FID = fid.compute_fid(Wasserstein_PQ_30, input_dir)
print("FID: "+str(FID))

##MMD PQ
MMD_PQ_5 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/mmd_vq_2_5"
MMD_PQ_10 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/mmd_vq_2_10"
MMD_PQ_15 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/mmd_vq_2_15"
MMD_PQ_20 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/mmd_vq_2_20"
MMD_PQ_25 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/mmd_vq_2_25"
MMD_PQ_30 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/mmd_vq_2_30"
print(MMD_PQ_5)
FID = fid.compute_fid(MMD_PQ_5, input_dir)
print("FID: "+str(FID))

print(MMD_PQ_10)
FID = fid.compute_fid(MMD_PQ_10, input_dir)
print("FID: "+str(FID))

print(MMD_PQ_15)
FID = fid.compute_fid(MMD_PQ_15, input_dir)
print("FID: "+str(FID))

print(MMD_PQ_20)
FID = fid.compute_fid(MMD_PQ_20, input_dir)
print("FID: "+str(FID))

print(MMD_PQ_25)
FID = fid.compute_fid(MMD_PQ_25, input_dir)
print("FID: "+str(FID))

print(MMD_PQ_30)
FID = fid.compute_fid(MMD_PQ_30, input_dir)
print("FID: "+str(FID))

print("#################SQ method###########################")
##FSQ
FSQ_5 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/fsq_8_4_5"
FSQ_10 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/fsq_8_4_10"
FSQ_15 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/fsq_8_4_15"
FSQ_20 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/fsq_8_4_20"
FSQ_25 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/fsq_8_4_25"
FSQ_30 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/fsq_8_4_30"
print(FSQ_5)
FID = fid.compute_fid(FSQ_5, input_dir)
print("FID: "+str(FID))

print(FSQ_10)
FID = fid.compute_fid(FSQ_10, input_dir)
print("FID: "+str(FID))

print(FSQ_15)
FID = fid.compute_fid(FSQ_15, input_dir)
print("FID: "+str(FID))

print(FSQ_20)
FID = fid.compute_fid(FSQ_20, input_dir)
print("FID: "+str(FID))

print(FSQ_25)
FID = fid.compute_fid(FSQ_25, input_dir)
print("FID: "+str(FID))

print(FSQ_30)
FID = fid.compute_fid(FSQ_30, input_dir)
print("FID: "+str(FID))

##BSQ
BSQ_5 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/bsq_16_2_5"
BSQ_10 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/bsq_16_2_10"
BSQ_15 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/bsq_16_2_15"
BSQ_20 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/bsq_16_2_20"
BSQ_25 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/bsq_16_2_25"
BSQ_30 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/bsq_16_2_30"
print(BSQ_5)
FID = fid.compute_fid(BSQ_5, input_dir)
print("FID: "+str(FID))

print(BSQ_10)
FID = fid.compute_fid(BSQ_10, input_dir)
print("FID: "+str(FID))

print(BSQ_15)
FID = fid.compute_fid(BSQ_15, input_dir)
print("FID: "+str(FID))

print(BSQ_20)
FID = fid.compute_fid(BSQ_20, input_dir)
print("FID: "+str(FID))

print(BSQ_25)
FID = fid.compute_fid(BSQ_25, input_dir)
print("FID: "+str(FID))

print(BSQ_30)
FID = fid.compute_fid(BSQ_30, input_dir)
print("FID: "+str(FID))

##LFQ
LFQ_5 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/lfq_16_2_5"
LFQ_10 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/lfq_16_2_10"
LFQ_15 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/lfq_16_2_15"
LFQ_20 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/lfq_16_2_20"
LFQ_25 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/lfq_16_2_25"
LFQ_30 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/reconstruction/CelebAHQ/lfq_16_2_30"
print(LFQ_5)
FID = fid.compute_fid(LFQ_5, input_dir)
print("FID: "+str(FID))

print(LFQ_10)
FID = fid.compute_fid(LFQ_10, input_dir)
print("FID: "+str(FID))

print(LFQ_15)
FID = fid.compute_fid(LFQ_15, input_dir)
print("FID: "+str(FID))

print(LFQ_20)
FID = fid.compute_fid(LFQ_20, input_dir)
print("FID: "+str(FID))

print(LFQ_25)
FID = fid.compute_fid(LFQ_25, input_dir)
print("FID: "+str(FID))

print(LFQ_30)
FID = fid.compute_fid(LFQ_30, input_dir)
print("FID: "+str(FID))

