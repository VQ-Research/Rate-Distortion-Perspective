#!/bin/bash
#SBATCH --job-name=fsq_celeba
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_h100_b2,gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --nodes=1
#SBATCH --mem=80gb
#SBATCH --cpus-per-task 10
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --output /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/slurm/CelebAHQ/fsq_celeba.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/slurm/CelebAHQ/fsq_celeba.err

module load gcc opencv/4.8.1
source /home/sunset/environment/VQ-Tokenizer/bin/activate
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=13582 train_SQ.py --VQ=fsq --dataset_name=CelebAHQ --global_batch_size=8  --L=4 --project_dim=8 --beta=1.0
