#!/bin/bash
#SBATCH --job-name=rFID_mmd_supplement_refinement
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task 10
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output /project/6105494/sunset/VQ-Projects/VQ-Transplant/metrics/Refinement/ImageNet/rFID_mmd_supplement_refinement.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/VQ-Transplant/metrics/Refinement/ImageNet/rFID_mmd_supplement_refinement.err

source /home/sunset/environment/FID/bin/activate
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant2/reconstruction/Refinement/ImageNet --sample_name mmd_vq_refinement_1024_False_1.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant2/reconstruction/Refinement/ImageNet --sample_name mmd_vq_refinement_1024_False_2.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant2/reconstruction/Refinement/ImageNet --sample_name mmd_vq_refinement_1024_False_3.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant2/reconstruction/Refinement/ImageNet --sample_name mmd_vq_refinement_1024_False_4.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant2/reconstruction/Refinement/ImageNet --sample_name mmd_vq_refinement_1024_False_5.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant2/reconstruction/Refinement/ImageNet --sample_name mmd_vq_refinement_2048_False_1.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant2/reconstruction/Refinement/ImageNet --sample_name mmd_vq_refinement_2048_False_2.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant2/reconstruction/Refinement/ImageNet --sample_name mmd_vq_refinement_2048_False_3.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant2/reconstruction/Refinement/ImageNet --sample_name mmd_vq_refinement_2048_False_4.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant2/reconstruction/Refinement/ImageNet --sample_name mmd_vq_refinement_2048_False_5.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant2/reconstruction/Refinement/ImageNet --sample_name mmd_vq_refinement_4096_False_1.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant2/reconstruction/Refinement/ImageNet --sample_name mmd_vq_refinement_4096_False_2.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant2/reconstruction/Refinement/ImageNet --sample_name mmd_vq_refinement_4096_False_3.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant2/reconstruction/Refinement/ImageNet --sample_name mmd_vq_refinement_4096_False_4.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant2/reconstruction/Refinement/ImageNet --sample_name mmd_vq_refinement_4096_False_5.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant2/reconstruction/Refinement/ImageNet --sample_name mmd_vq_refinement_8192_False_1.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant2/reconstruction/Refinement/ImageNet --sample_name mmd_vq_refinement_8192_False_2.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant2/reconstruction/Refinement/ImageNet --sample_name mmd_vq_refinement_8192_False_3.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant2/reconstruction/Refinement/ImageNet --sample_name mmd_vq_refinement_8192_False_4.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant2/reconstruction/Refinement/ImageNet --sample_name mmd_vq_refinement_8192_False_5.npz