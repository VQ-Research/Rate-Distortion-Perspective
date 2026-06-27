#!/bin/bash
#SBATCH --job-name=rFID_mmd_supplement_transplant
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task 10
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output /project/6105494/sunset/VQ-Projects/VQ-Transplant/metrics/Transplant/ImageNet/rFID_mmd_supplement_transplant.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/VQ-Transplant/metrics/Transplant/ImageNet/rFID_mmd_supplement_transplant.err

source /home/sunset/environment/FID/bin/activate
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant2/reconstruction/Transplant/ImageNet --sample_name mmd_vq_transplant_1024_False.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant2/reconstruction/Transplant/ImageNet --sample_name mmd_vq_transplant_2048_False.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant2/reconstruction/Transplant/ImageNet --sample_name mmd_vq_transplant_4096_False.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant2/reconstruction/Transplant/ImageNet --sample_name mmd_vq_transplant_8192_False.npz
