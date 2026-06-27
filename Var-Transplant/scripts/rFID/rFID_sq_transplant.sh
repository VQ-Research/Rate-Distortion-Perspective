#!/bin/bash
#SBATCH --job-name=rfid_sq_transplant
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task 10
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output /project/6105494/sunset/VQ-Projects/VQ-Transplant/metrics/Transplant/ImageNet/rfid_sq_transplant.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/VQ-Transplant/metrics/Transplant/ImageNet/rfid_sq_transplant.err

source /home/sunset/environment/FID/bin/activate
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant/reconstruction/Transplant/ImageNet --sample_name fsq_transplant_8_4.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant/reconstruction/Transplant/ImageNet --sample_name lfq_transplant_16_2.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant/reconstruction/Transplant/ImageNet --sample_name bsq_transplant_16_2.npz