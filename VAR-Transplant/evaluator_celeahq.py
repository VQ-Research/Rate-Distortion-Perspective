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

print("#################refinement-stage###########################")
vanilla_vq_refinement_5 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/vanilla_vq_refinement_65536_5"
vanilla_vq_refinement_10 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/vanilla_vq_refinement_65536_10"
vanilla_vq_refinement_15 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/vanilla_vq_refinement_65536_15"
vanilla_vq_refinement_20 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/vanilla_vq_refinement_65536_20"
vanilla_vq_refinement_25 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/vanilla_vq_refinement_65536_25"
vanilla_vq_refinement_30 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/vanilla_vq_refinement_65536_30"

ema_vq_refinement_5 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/ema_vq_refinement_65536_5"
ema_vq_refinement_10 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/ema_vq_refinement_65536_10"
ema_vq_refinement_15 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/ema_vq_refinement_65536_15"
ema_vq_refinement_20 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/ema_vq_refinement_65536_20"
ema_vq_refinement_25 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/ema_vq_refinement_65536_25"
ema_vq_refinement_30 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/ema_vq_refinement_65536_30"

online_vq_refinement_5 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/online_vq_refinement_65536_5"
online_vq_refinement_10 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/online_vq_refinement_65536_10"
online_vq_refinement_15 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/online_vq_refinement_65536_15"
online_vq_refinement_20 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/online_vq_refinement_65536_20"
online_vq_refinement_25 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/online_vq_refinement_65536_25"
online_vq_refinement_30 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/online_vq_refinement_65536_30"

wasserstein_vq_refinement_5 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/wasserstein_vq_refinement_65536_5"
wasserstein_vq_refinement_10 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/wasserstein_vq_refinement_65536_10"
wasserstein_vq_refinement_15 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/wasserstein_vq_refinement_65536_15"
wasserstein_vq_refinement_20 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/wasserstein_vq_refinement_65536_20"
wasserstein_vq_refinement_25 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/wasserstein_vq_refinement_65536_25"
wasserstein_vq_refinement_30 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/wasserstein_vq_refinement_65536_30"

mmd_vq_refinement_5 ="/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/mmd_vq_refinement_65536_5"
mmd_vq_refinement_10 ="/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/mmd_vq_refinement_65536_10"
mmd_vq_refinement_15 ="/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/mmd_vq_refinement_65536_15"
mmd_vq_refinement_20 ="/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/mmd_vq_refinement_65536_20"
mmd_vq_refinement_25 ="/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/mmd_vq_refinement_65536_25"
mmd_vq_refinement_30 ="/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/mmd_vq_refinement_65536_30"

vanilla_pq_refinement_5 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/vanilla_vq_refinement_2_5"
vanilla_pq_refinement_10 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/vanilla_vq_refinement_2_10"
vanilla_pq_refinement_15 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/vanilla_vq_refinement_2_15"
vanilla_pq_refinement_20 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/vanilla_vq_refinement_2_20"
vanilla_pq_refinement_25 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/vanilla_vq_refinement_2_25"
vanilla_pq_refinement_30 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/vanilla_vq_refinement_2_30"

ema_pq_refinement_5 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/ema_vq_refinement_2_5"
ema_pq_refinement_10 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/ema_vq_refinement_2_10"
ema_pq_refinement_15 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/ema_vq_refinement_2_15"
ema_pq_refinement_20 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/ema_vq_refinement_2_20"
ema_pq_refinement_25 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/ema_vq_refinement_2_25"
ema_pq_refinement_30 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/ema_vq_refinement_2_30"

online_pq_refinement_5 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/online_vq_refinement_2_5"
online_pq_refinement_10 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/online_vq_refinement_2_10"
online_pq_refinement_15 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/online_vq_refinement_2_15"
online_pq_refinement_20 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/online_vq_refinement_2_20"
online_pq_refinement_25 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/online_vq_refinement_2_25"
online_pq_refinement_30 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/online_vq_refinement_2_30"

wasserstein_pq_refinement_5 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/wasserstein_vq_refinement_2_5"
wasserstein_pq_refinement_10 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/wasserstein_vq_refinement_2_10"
wasserstein_pq_refinement_15 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/wasserstein_vq_refinement_2_15"
wasserstein_pq_refinement_20 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/wasserstein_vq_refinement_2_20"
wasserstein_pq_refinement_25 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/wasserstein_vq_refinement_2_25"
wasserstein_pq_refinement_30 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/wasserstein_vq_refinement_2_30"

mmd_pq_refinement_5 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/mmd_vq_refinement_2_5"
mmd_pq_refinement_10 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/mmd_vq_refinement_2_10"
mmd_pq_refinement_15 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/mmd_vq_refinement_2_15"
mmd_pq_refinement_20 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/mmd_vq_refinement_2_20"
mmd_pq_refinement_25 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/mmd_vq_refinement_2_25"
mmd_pq_refinement_30 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/mmd_vq_refinement_2_30"

fsq_refinement_5 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/fsq_refinement_8_4_5"
fsq_refinement_10 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/fsq_refinement_8_4_10"
fsq_refinement_15 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/fsq_refinement_8_4_15"
fsq_refinement_20 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/fsq_refinement_8_4_20"
fsq_refinement_25 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/fsq_refinement_8_4_25"
fsq_refinement_30 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/fsq_refinement_8_4_30"

bsq_refinement_5 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/bsq_refinement_16_2_5"
bsq_refinement_10 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/bsq_refinement_16_2_10"
bsq_refinement_15 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/bsq_refinement_16_2_15"
bsq_refinement_20 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/bsq_refinement_16_2_20"
bsq_refinement_25 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/bsq_refinement_16_2_25"
bsq_refinement_30 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/bsq_refinement_16_2_30"

lfq_refinement_5 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/lfq_refinement_16_2_5"
lfq_refinement_10 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/lfq_refinement_16_2_10"
lfq_refinement_15 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/lfq_refinement_16_2_15"
lfq_refinement_20 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/lfq_refinement_16_2_20"
lfq_refinement_25 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/lfq_refinement_16_2_25"
lfq_refinement_30 = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/lfq_refinement_16_2_30"


#####################################
print(vanilla_vq_refinement_5)
FID = fid.compute_fid(vanilla_vq_refinement_5, input_dir)
print("FID: "+str(FID))

print(vanilla_vq_refinement_10)
FID = fid.compute_fid(vanilla_vq_refinement_10, input_dir)
print("FID: "+str(FID))

print(vanilla_vq_refinement_15)
FID = fid.compute_fid(vanilla_vq_refinement_15, input_dir)
print("FID: "+str(FID))

print(vanilla_vq_refinement_20)
FID = fid.compute_fid(vanilla_vq_refinement_20, input_dir)
print("FID: "+str(FID))

print(vanilla_vq_refinement_25)
FID = fid.compute_fid(vanilla_vq_refinement_25, input_dir)
print("FID: "+str(FID))

print(vanilla_vq_refinement_30)
FID = fid.compute_fid(vanilla_vq_refinement_30, input_dir)
print("FID: "+str(FID))

####################################

print(ema_vq_refinement_5)
FID = fid.compute_fid(ema_vq_refinement_5, input_dir)
print("FID: "+str(FID))

print(ema_vq_refinement_10)
FID = fid.compute_fid(ema_vq_refinement_10, input_dir)
print("FID: "+str(FID))

print(ema_vq_refinement_15)
FID = fid.compute_fid(ema_vq_refinement_15, input_dir)
print("FID: "+str(FID))

print(ema_vq_refinement_20)
FID = fid.compute_fid(ema_vq_refinement_20, input_dir)
print("FID: "+str(FID))

print(ema_vq_refinement_25)
FID = fid.compute_fid(ema_vq_refinement_25, input_dir)
print("FID: "+str(FID))

print(ema_vq_refinement_30)
FID = fid.compute_fid(ema_vq_refinement_30, input_dir)
print("FID: "+str(FID))
####################################

print(online_vq_refinement_5)
FID = fid.compute_fid(online_vq_refinement_5, input_dir)
print("FID: "+str(FID))

print(online_vq_refinement_10)
FID = fid.compute_fid(online_vq_refinement_10, input_dir)
print("FID: "+str(FID))

print(online_vq_refinement_15)
FID = fid.compute_fid(online_vq_refinement_15, input_dir)
print("FID: "+str(FID))

print(online_vq_refinement_20)
FID = fid.compute_fid(online_vq_refinement_20, input_dir)
print("FID: "+str(FID))

print(online_vq_refinement_25)
FID = fid.compute_fid(online_vq_refinement_25, input_dir)
print("FID: "+str(FID))

print(online_vq_refinement_30)
FID = fid.compute_fid(online_vq_refinement_30, input_dir)
print("FID: "+str(FID))

####################################

print(wasserstein_vq_refinement_5)
FID = fid.compute_fid(wasserstein_vq_refinement_5, input_dir)
print("FID: "+str(FID))

print(wasserstein_vq_refinement_10)
FID = fid.compute_fid(wasserstein_vq_refinement_10, input_dir)
print("FID: "+str(FID))

print(wasserstein_vq_refinement_15)
FID = fid.compute_fid(wasserstein_vq_refinement_15, input_dir)
print("FID: "+str(FID))

print(wasserstein_vq_refinement_20)
FID = fid.compute_fid(wasserstein_vq_refinement_20, input_dir)
print("FID: "+str(FID))

print(wasserstein_vq_refinement_25)
FID = fid.compute_fid(wasserstein_vq_refinement_25, input_dir)
print("FID: "+str(FID))

print(wasserstein_vq_refinement_30)
FID = fid.compute_fid(wasserstein_vq_refinement_30, input_dir)
print("FID: "+str(FID))

####################################

print(mmd_vq_refinement_5)
FID = fid.compute_fid(mmd_vq_refinement_5, input_dir)
print("FID: "+str(FID))

print(mmd_vq_refinement_10)
FID = fid.compute_fid(mmd_vq_refinement_10, input_dir)
print("FID: "+str(FID))

print(mmd_vq_refinement_15)
FID = fid.compute_fid(mmd_vq_refinement_15, input_dir)
print("FID: "+str(FID))

print(mmd_vq_refinement_20)
FID = fid.compute_fid(mmd_vq_refinement_20, input_dir)
print("FID: "+str(FID))

print(mmd_vq_refinement_25)
FID = fid.compute_fid(mmd_vq_refinement_25, input_dir)
print("FID: "+str(FID))

print(mmd_vq_refinement_30)
FID = fid.compute_fid(mmd_vq_refinement_30, input_dir)
print("FID: "+str(FID))

####################################

print(vanilla_pq_refinement_5)
FID = fid.compute_fid(vanilla_pq_refinement_5, input_dir)
print("FID: "+str(FID))

print(vanilla_pq_refinement_10)
FID = fid.compute_fid(vanilla_pq_refinement_10, input_dir)
print("FID: "+str(FID))

print(vanilla_pq_refinement_15)
FID = fid.compute_fid(vanilla_pq_refinement_15, input_dir)
print("FID: "+str(FID))

print(vanilla_pq_refinement_20)
FID = fid.compute_fid(vanilla_pq_refinement_20, input_dir)
print("FID: "+str(FID))

print(vanilla_pq_refinement_25)
FID = fid.compute_fid(vanilla_pq_refinement_25, input_dir)
print("FID: "+str(FID))

print(vanilla_pq_refinement_30)
FID = fid.compute_fid(vanilla_pq_refinement_30, input_dir)
print("FID: "+str(FID))

####################################

print(ema_pq_refinement_5)
FID = fid.compute_fid(ema_pq_refinement_5, input_dir)
print("FID: "+str(FID))

print(ema_pq_refinement_10)
FID = fid.compute_fid(ema_pq_refinement_10, input_dir)
print("FID: "+str(FID))

print(ema_pq_refinement_15)
FID = fid.compute_fid(ema_pq_refinement_15, input_dir)
print("FID: "+str(FID))

print(ema_pq_refinement_20)
FID = fid.compute_fid(ema_pq_refinement_20, input_dir)
print("FID: "+str(FID))

print(ema_pq_refinement_25)
FID = fid.compute_fid(ema_pq_refinement_25, input_dir)
print("FID: "+str(FID))

print(ema_pq_refinement_30)
FID = fid.compute_fid(ema_pq_refinement_30, input_dir)
print("FID: "+str(FID))

####################################

print(online_pq_refinement_5)
FID = fid.compute_fid(online_pq_refinement_5, input_dir)
print("FID: "+str(FID))

print(online_pq_refinement_10)
FID = fid.compute_fid(online_pq_refinement_10, input_dir)
print("FID: "+str(FID))

print(online_pq_refinement_15)
FID = fid.compute_fid(online_pq_refinement_15, input_dir)
print("FID: "+str(FID))

print(online_pq_refinement_20)
FID = fid.compute_fid(online_pq_refinement_20, input_dir)
print("FID: "+str(FID))

print(online_pq_refinement_25)
FID = fid.compute_fid(online_pq_refinement_25, input_dir)
print("FID: "+str(FID))

print(online_pq_refinement_30)
FID = fid.compute_fid(online_pq_refinement_30, input_dir)
print("FID: "+str(FID))

####################################

print(wasserstein_pq_refinement_5)
FID = fid.compute_fid(wasserstein_pq_refinement_5, input_dir)
print("FID: "+str(FID))

print(wasserstein_pq_refinement_10)
FID = fid.compute_fid(wasserstein_pq_refinement_10, input_dir)
print("FID: "+str(FID))

print(wasserstein_pq_refinement_15)
FID = fid.compute_fid(wasserstein_pq_refinement_15, input_dir)
print("FID: "+str(FID))

print(wasserstein_pq_refinement_20)
FID = fid.compute_fid(wasserstein_pq_refinement_20, input_dir)
print("FID: "+str(FID))

print(wasserstein_pq_refinement_25)
FID = fid.compute_fid(wasserstein_pq_refinement_25, input_dir)
print("FID: "+str(FID))

print(wasserstein_pq_refinement_30)
FID = fid.compute_fid(wasserstein_pq_refinement_30, input_dir)
print("FID: "+str(FID))


####################################

print(mmd_pq_refinement_5)
FID = fid.compute_fid(mmd_pq_refinement_5, input_dir)
print("FID: "+str(FID))

print(mmd_pq_refinement_10)
FID = fid.compute_fid(mmd_pq_refinement_10, input_dir)
print("FID: "+str(FID))

print(mmd_pq_refinement_15)
FID = fid.compute_fid(mmd_pq_refinement_15, input_dir)
print("FID: "+str(FID))

print(mmd_pq_refinement_20)
FID = fid.compute_fid(mmd_pq_refinement_20, input_dir)
print("FID: "+str(FID))

print(mmd_pq_refinement_25)
FID = fid.compute_fid(mmd_pq_refinement_25, input_dir)
print("FID: "+str(FID))

print(mmd_pq_refinement_30)
FID = fid.compute_fid(mmd_pq_refinement_30, input_dir)
print("FID: "+str(FID))

####################################

print(fsq_refinement_5)
FID = fid.compute_fid(fsq_refinement_5, input_dir)
print("FID: "+str(FID))

print(fsq_refinement_10)
FID = fid.compute_fid(fsq_refinement_10, input_dir)
print("FID: "+str(FID))

print(fsq_refinement_15)
FID = fid.compute_fid(fsq_refinement_15, input_dir)
print("FID: "+str(FID))

print(fsq_refinement_20)
FID = fid.compute_fid(fsq_refinement_20, input_dir)
print("FID: "+str(FID))

print(fsq_refinement_25)
FID = fid.compute_fid(fsq_refinement_25, input_dir)
print("FID: "+str(FID))

print(fsq_refinement_30)
FID = fid.compute_fid(fsq_refinement_30, input_dir)
print("FID: "+str(FID))

####################################

print(bsq_refinement_5)
FID = fid.compute_fid(bsq_refinement_5, input_dir)
print("FID: "+str(FID))

print(bsq_refinement_10)
FID = fid.compute_fid(bsq_refinement_10, input_dir)
print("FID: "+str(FID))

print(bsq_refinement_15)
FID = fid.compute_fid(bsq_refinement_15, input_dir)
print("FID: "+str(FID))

print(bsq_refinement_20)
FID = fid.compute_fid(bsq_refinement_20, input_dir)
print("FID: "+str(FID))

print(bsq_refinement_25)
FID = fid.compute_fid(bsq_refinement_25, input_dir)
print("FID: "+str(FID))

print(bsq_refinement_30)
FID = fid.compute_fid(bsq_refinement_30, input_dir)
print("FID: "+str(FID))

####################################

print(lfq_refinement_5)
FID = fid.compute_fid(lfq_refinement_5, input_dir)
print("FID: "+str(FID))

print(lfq_refinement_10)
FID = fid.compute_fid(lfq_refinement_10, input_dir)
print("FID: "+str(FID))

print(lfq_refinement_15)
FID = fid.compute_fid(lfq_refinement_15, input_dir)
print("FID: "+str(FID))

print(lfq_refinement_20)
FID = fid.compute_fid(lfq_refinement_20, input_dir)
print("FID: "+str(FID))

print(lfq_refinement_25)
FID = fid.compute_fid(lfq_refinement_25, input_dir)
print("FID: "+str(FID))

print(lfq_refinement_30)
FID = fid.compute_fid(lfq_refinement_30, input_dir)
print("FID: "+str(FID))




