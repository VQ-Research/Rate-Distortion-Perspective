#!/bin/bash
#SBATCH --job-name=ema_pq_refinement
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task 10
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --output /project/6105494/sunset/VQ-Projects/VQ-Transplant/slurm/Refinement/ImageNet/ema_pq_refinement.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/VQ-Transplant/slurm/Refinement/ImageNet/ema_pq_refinement.err

module load gcc opencv/4.8.1
source /home/sunset/environment/VQ-Tokenizer/bin/activate
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=12953 train_refinement.py --VQ=ema_vq --dataset_name=ImageNet --global_batch_size=64 --codebook_size 256  --codebook_dim=8 --pq=2 --stage=refinement --alpha=1.0 --beta=1.0 --gamma=0.0 --checkpoint_name checkpoint-ema_vq_transplant_False_ImageNet_model_256_8_2_2_loss_1.0_1.0_0.0_0.4.pth.tar
