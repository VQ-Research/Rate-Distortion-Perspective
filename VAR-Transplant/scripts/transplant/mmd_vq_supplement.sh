#!/bin/bash
#SBATCH --job-name=mmd_vq_supplement
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task 10
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --output /project/6105494/sunset/VQ-Projects/VQ-Transplant2/slurm/Transplant/ImageNet/mmd_vq_supplement.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/VQ-Transplant2/slurm/Transplant/ImageNet/mmd_vq_supplement.err


module load gcc opencv/4.8.1
source /home/sunset/environment/VQ-Tokenizer/bin/activate
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=19255 train_VQ_transplant.py --VQ=mmd_vq --dataset_name=ImageNet --path=vector --global_batch_size=64 --codebook_size 1024  --codebook_dim=16 --stage=transplant --alpha=1.0 --beta=1.0 --gamma=1.0
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=19255 train_VQ_transplant.py --VQ=mmd_vq --dataset_name=ImageNet --path=vector --global_batch_size=64 --codebook_size 2048  --codebook_dim=16 --stage=transplant --alpha=1.0 --beta=1.0 --gamma=1.0
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=19255 train_VQ_transplant.py --VQ=mmd_vq --dataset_name=ImageNet --path=vector --global_batch_size=64 --codebook_size 4096  --codebook_dim=16 --stage=transplant --alpha=1.0 --beta=1.0 --gamma=1.0
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=19255 train_VQ_transplant.py --VQ=mmd_vq --dataset_name=ImageNet --path=vector --global_batch_size=64 --codebook_size 8192  --codebook_dim=16 --stage=transplant --alpha=1.0 --beta=1.0 --gamma=1.0