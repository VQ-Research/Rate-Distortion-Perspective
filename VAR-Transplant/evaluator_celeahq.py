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
vanilla_vq_refinement = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/vanilla_vq_refinement_65536"
ema_vq_refinement = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/ema_vq_refinement_65536"
online_vq_refinement = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/online_vq_refinement_65536"
wasserstein_vq_refinement = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/wasserstein_vq_refinement_65536"
mmd_vq_refinement ="/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/mmd_vq_refinement_65536"

vanilla_pq_refinement = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/vanilla_vq_refinement_2"
ema_pq_refinement = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/ema_vq_refinement_2"
online_pq_refinement = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/online_vq_refinement_2"
wasserstein_pq_refinement = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/wasserstein_vq_refinement_2"
mmd_pq_refinement = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/mmd_vq_refinement_2"

fsq_refinement = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/fsq_refinement_8_4"
bsq_refinement = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/bsq_refinement_16_2"
lfq_refinement = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/CelebAHQ/lfq_refinement_16_2"

print(vanilla_vq_refinement)
FID = fid.compute_fid(vanilla_vq_refinement, input_dir)
print("FID: "+str(FID))

print(ema_vq_refinement)
FID = fid.compute_fid(ema_vq_refinement, input_dir)
print("FID: "+str(FID))

print(online_vq_refinement)
FID = fid.compute_fid(online_vq_refinement, input_dir)
print("FID: "+str(FID))

print(wasserstein_vq_refinement)
FID = fid.compute_fid(wasserstein_vq_refinement, input_dir)
print("FID: "+str(FID))

print(mmd_vq_refinement)
FID = fid.compute_fid(mmd_vq_refinement, input_dir)
print("FID: "+str(FID))

print(vanilla_pq_refinement)
FID = fid.compute_fid(vanilla_pq_refinement, input_dir)
print("FID: "+str(FID))

print(ema_pq_refinement)
FID = fid.compute_fid(ema_pq_refinement, input_dir)
print("FID: "+str(FID))

print(online_pq_refinement)
FID = fid.compute_fid(online_pq_refinement, input_dir)
print("FID: "+str(FID))

print(wasserstein_pq_refinement)
FID = fid.compute_fid(wasserstein_pq_refinement, input_dir)
print("FID: "+str(FID))

print(mmd_pq_refinement)
FID = fid.compute_fid(mmd_pq_refinement, input_dir)
print("FID: "+str(FID))

print(fsq_refinement)
FID = fid.compute_fid(fsq_refinement, input_dir)
print("FID: "+str(FID))

print(bsq_refinement)
FID = fid.compute_fid(bsq_refinement, input_dir)
print("FID: "+str(FID))

print(lfq_refinement)
FID = fid.compute_fid(lfq_refinement, input_dir)
print("FID: "+str(FID))


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



