#!/bin/bash
#SBATCH --job-name=wasserstein_vq_celeba
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_h100_b2,gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --nodes=1
#SBATCH --mem=80gb
#SBATCH --cpus-per-task 10
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --output /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/slurm/CelebAHQ/wasserstein_vq_celeba.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/Pixel-Space/slurm/CelebAHQ/wasserstein_vq_celeba.err

module load gcc opencv/4.8.1
source /home/sunset/environment/VQ-Tokenizer/bin/activate
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=13595 train_VQ.py --VQ=wasserstein_vq --dataset_name=CelebAHQ --global_batch_size=16 --codebook_size=65536  --codebook_dim=16 --pq=1 --alpha=1.0 --beta=0.2 --gamma=0.5
