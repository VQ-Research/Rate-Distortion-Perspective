#!/bin/bash
#SBATCH --job-name=rFID_ffhq
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_l40s_b1,gpubase_l40s_b2,gpubase_l40s_b3
#SBATCH --nodes=1
#SBATCH --mem=80gb
#SBATCH --cpus-per-task 10
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/metrics/rFID_ffhq.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/metrics/rFID_ffhq.err

module load gcc opencv/4.8.1
source /home/sunset/environment/VQ-Tokenizer/bin/activate
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 --master_port=15582 evaluator_ffhq.py