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

input_dir = "/project/6105494/shared/reconstruction/FFHQ"

print("#################refinement-stage###########################")
vanilla_vq_refinement = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/FFHQ/vanilla_vq_refinement_65536"
ema_vq_refinement = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/FFHQ/ema_vq_refinement_65536"
online_vq_refinement = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/FFHQ/online_vq_refinement_65536"
wasserstein_vq_refinement = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/FFHQ/wasserstein_vq_refinement_65536"
mmd_vq_refinement ="/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/FFHQ/mmd_vq_refinement_65536"

vanilla_pq_refinement = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/FFHQ/vanilla_vq_refinement_2"
ema_pq_refinement = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/FFHQ/ema_vq_refinement_2"
online_pq_refinement = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/FFHQ/online_vq_refinement_2"
wasserstein_pq_refinement = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/FFHQ/wasserstein_vq_refinement_2"
mmd_pq_refinement = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/FFHQ/mmd_vq_refinement_2"

fsq_refinement = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/FFHQ/fsq_refinement_8_4"
bsq_refinement = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/FFHQ/bsq_refinement_16_2"
lfq_refinement = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/FFHQ/lfq_refinement_16_2"

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
vanilla_vq_transplant = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/FFHQ/vanilla_vq_transplant_65536"
ema_vq_transplant = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/FFHQ/ema_vq_transplant_65536"
online_vq_transplant = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/FFHQ/online_vq_transplant_65536"
wasserstein_vq_transplant = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/FFHQ/wasserstein_vq_transplant_65536"

vanilla_pq_transplant = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/FFHQ/vanilla_vq_transplant_2"
ema_pq_transplant = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/FFHQ/ema_vq_transplant_2"
online_pq_transplant = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/FFHQ/online_vq_transplant_2"
wasserstein_pq_transplant = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/FFHQ/wasserstein_vq_transplant_2"
mmd_pq_transplant = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/FFHQ/mmd_vq_transplant_2"

fsq_transplant = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/FFHQ/fsq_transplant_8_4"
bsq_transplant = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/FFHQ/bsq_transplant_16_2"
lfq_transplant = "/project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/FFHQ/lfq_transplant_16_2"

print(vanilla_vq_transplant)
FID = fid.compute_fid(vanilla_vq_transplant, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(ema_vq_transplant)
FID = fid.compute_fid(ema_vq_transplant, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(online_vq_transplant)
FID = fid.compute_fid(online_vq_transplant, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_vq_transplant)
FID = fid.compute_fid(wasserstein_vq_transplant, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(vanilla_pq_transplant)
FID = fid.compute_fid(vanilla_pq_transplant, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(ema_pq_transplant)
FID = fid.compute_fid(ema_pq_transplant, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(online_pq_transplant)
FID = fid.compute_fid(online_pq_transplant, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_pq_transplant)
FID = fid.compute_fid(wasserstein_pq_transplant, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(mmd_pq_transplant)
FID = fid.compute_fid(mmd_pq_transplant, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(fsq_transplant)
FID = fid.compute_fid(fsq_transplant, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(bsq_transplant)
FID = fid.compute_fid(bsq_transplant, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(lfq_transplant)
FID = fid.compute_fid(lfq_transplant, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))





