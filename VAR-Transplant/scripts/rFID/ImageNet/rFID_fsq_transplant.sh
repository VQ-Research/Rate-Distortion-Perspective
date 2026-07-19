#!/bin/bash
#SBATCH --job-name=rfid_fsq_transplant
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_l40s_b1,gpubase_l40s_b2,gpubase_l40s_b3,gpubase_l40s_b4,gpubase_l40s_b5
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task 10
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/metrics/rfid_fsq_transplant.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/metrics/rfid_fsq_transplant.err

source /home/sunset/environment/FID/bin/activate
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Transplant/ImageNet --sample_name fsq_transplant_8_4.npz
