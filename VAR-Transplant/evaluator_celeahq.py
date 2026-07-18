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

print("#################transplant-stage###########################")
vanilla_vq_transplant = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/CelebAHQ/vanilla_vq_transplant_65536"
ema_vq_transplant = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/CelebAHQ/ema_vq_transplant_65536"
online_vq_transplant = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/CelebAHQ/online_vq_transplant_65536"
wasserstein_vq_transplant = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/CelebAHQ/wasserstein_vq_transplant_65536"

vanilla_pq_transplant = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/CelebAHQ/vanilla_vq_transplant_2"
ema_pq_transplant = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/CelebAHQ/ema_vq_transplant_2"
online_pq_transplant = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/CelebAHQ/online_vq_transplant_2"
wasserstein_pq_transplant = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/CelebAHQ/wasserstein_vq_transplant_2"
mmd_pq_transplant = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/CelebAHQ/mmd_vq_transplant_2"

fsq_transplant = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/CelebAHQ/fsq_transplant_8_4"
bsq_transplant = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/CelebAHQ/bsq_transplant_16_2"
lfq_transplant = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/CelebAHQ/lfq_transplant_16_2"

print(vanilla_vq_transplant)
FID = fid.compute_fid(vanilla_vq_transplant, input_dir)
print("FID: "+str(FID))

print(ema_vq_transplant)
FID = fid.compute_fid(ema_vq_transplant, input_dir)
print("FID: "+str(FID))

print(online_vq_transplant)
FID = fid.compute_fid(online_vq_transplant, input_dir)
print("FID: "+str(FID))

print(wasserstein_vq_transplant)
FID = fid.compute_fid(wasserstein_vq_transplant, input_dir)
print("FID: "+str(FID))

print(vanilla_pq_transplant)
FID = fid.compute_fid(vanilla_pq_transplant, input_dir)
print("FID: "+str(FID))

print(ema_pq_transplant)
FID = fid.compute_fid(ema_pq_transplant, input_dir)
print("FID: "+str(FID))

print(online_pq_transplant)
FID = fid.compute_fid(online_pq_transplant, input_dir)
print("FID: "+str(FID))

print(wasserstein_pq_transplant)
FID = fid.compute_fid(wasserstein_pq_transplant, input_dir)
print("FID: "+str(FID))

print(mmd_pq_transplant)
FID = fid.compute_fid(mmd_pq_transplant, input_dir)
print("FID: "+str(FID))

print(fsq_transplant)
FID = fid.compute_fid(fsq_transplant, input_dir)
print("FID: "+str(FID))

print(bsq_transplant)
FID = fid.compute_fid(bsq_transplant, input_dir)
print("FID: "+str(FID))

print(lfq_transplant)
FID = fid.compute_fid(lfq_transplant, input_dir)
print("FID: "+str(FID))

