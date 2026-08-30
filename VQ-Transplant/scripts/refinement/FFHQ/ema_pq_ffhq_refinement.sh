#!/bin/bash
#SBATCH --job-name=ema_pq_ffhq_refinement
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --nodes=1
#SBATCH --mem=80gb
#SBATCH --cpus-per-task 5
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --output /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/slurm/Refinement/FFHQ/ema_pq_ffhq_refinement.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/slurm/Refinement/FFHQ/ema_pq_ffhq_refinement.err

module load gcc opencv/4.8.1
source /home/sunset/environment/VQ-Tokenizer/bin/activate
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=25584 train_refinement.py --VQ=ema_vq --dataset_name=FFHQ --global_batch_size=64 --codebook_size=256  --codebook_dim=8 --pq=2 --stage=refinement --alpha=1.0 --beta=1.0 --gamma=0.0 --checkpoint_name=checkpoint-ema_vq_transplant_FFHQ_model_256_8_2_loss_1.0_1.0_0.0_0.4.pth.tar
