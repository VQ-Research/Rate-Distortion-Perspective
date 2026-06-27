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

input_dir = "/projects/yuanai/projects/VQ-Transplant3/reconstruction/CelebAHQ"

mmd_transplant_16384 = "/projects/yuanai/projects/VQ-Transplant3/reconstruction/Transplant/CelebAHQ/mmd_vq_transplant_16384_False"
wasserstein_transplant_16384 = "/projects/yuanai/projects/VQ-Transplant3/reconstruction/Transplant/CelebAHQ/wasserstein_vq_transplant_16384_False"

mmd_refinement_16384 = "/projects/yuanai/projects/VQ-Transplant3/reconstruction/Refinement/CelebAHQ/mmd_vq_refinement_16384_False"
wasserstein_refinement_16384 = "/projects/yuanai/projects/VQ-Transplant3/reconstruction/Refinement/CelebAHQ/wasserstein_vq_refinement_16384_False"


print("#################transplant-stage###########################")
print(mmd_transplant_16384)
FID = fid.compute_fid(mmd_transplant_16384, input_dir)
print("FID: "+str(FID))

print(wasserstein_transplant_16384)
FID = fid.compute_fid(wasserstein_transplant_16384, input_dir)
print("FID: "+str(FID))

print("#################Refinement-stage###########################")
print(mmd_refinement_16384)
FID = fid.compute_fid(mmd_refinement_16384, input_dir)
print("FID: "+str(FID))

print(wasserstein_refinement_16384)
FID = fid.compute_fid(wasserstein_refinement_16384, input_dir)
print("FID: "+str(FID))
