#!/bin/bash
#SBATCH --job-name=mmd_pq_ffhq_refinement
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_h100_b2,gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --nodes=1
#SBATCH --mem=80gb
#SBATCH --cpus-per-task 10
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --output /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/slurm/Refinement/FFHQ/mmd_pq_ffhq_refinement.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/slurm/Refinement/FFHQ/mmd_pq_ffhq_refinement.err

module load gcc opencv/4.8.1
source /home/sunset/environment/VQ-Tokenizer/bin/activate
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=25585 train_refinement.py --VQ=mmd_vq --dataset_name=FFHQ --global_batch_size=64 --codebook_size=256  --codebook_dim=8 --pq=2 --stage=refinement --alpha=1.0 --beta=0.2 --gamma=0.5 --checkpoint_name=checkpoint-mmd_vq_transplant_FFHQ_model_256_8_2_loss_1.0_0.2_0.5_0.4.pth.tar
