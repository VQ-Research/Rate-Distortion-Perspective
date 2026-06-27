#!/bin/bash
#SBATCH --job-name=mmd_vq_chunk8_refinement
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task 10
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --output /project/6105494/sunset/VQ-Projects/VQ-Transplant/slurm/Refinement/ImageNet/mmd_vq_chunk8_refinement.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/VQ-Transplant/slurm/Refinement/ImageNet/mmd_vq_chunk8_refinement.err

module load gcc opencv/4.8.1
source /home/sunset/environment/VQ-Tokenizer/bin/activate
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=16953 train_refinement.py --VQ=mmd_vq --dataset_name=ImageNet --global_batch_size=64 --codebook_size 16384 --codebook_dim=4 --chunks=8 --stage=refinement --alpha=1.0 --beta=0.2 --gamma=1.0 --checkpoint_name checkpoint-mmd_vq_transplant_False_ImageNet_model_16384_4_8_1_loss_1.0_0.2_1.0_0.4.pth.tar
