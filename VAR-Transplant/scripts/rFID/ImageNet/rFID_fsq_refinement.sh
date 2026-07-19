#!/bin/bash
#SBATCH --job-name=rfid_fsq_refinement
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_l40s_b1,gpubase_l40s_b2,gpubase_l40s_b3,gpubase_l40s_b4,gpubase_l40s_b5
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task 10
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/metrics/rfid_fsq_refinement.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/metrics/rfid_fsq_refinement.err

source /home/sunset/environment/FID/bin/activate
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/ImageNet --sample_name fsq_refinement_8_4_1.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/ImageNet --sample_name fsq_refinement_8_4_2.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/ImageNet --sample_name fsq_refinement_8_4_3.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/ImageNet --sample_name fsq_refinement_8_4_4.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/reconstruction/Refinement/ImageNet --sample_name fsq_refinement_8_4_5.npz