#!/bin/bash
#SBATCH --job-name=lfq_refinement
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task=10
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --output /project/6105494/sunset/VQ-Projects/VQ-Transplant/slurm/Refinement/ImageNet/lfq_refinement.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/VQ-Transplant/slurm/Refinement/ImageNet/lfq_refinement.err

module load gcc opencv/4.8.1
source /home/sunset/environment/VQ-Tokenizer/bin/activate
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=13258 train_refinement.py --VQ=lfq --dataset_name=ImageNet --global_batch_size=64  --L=2 --project_dim=16 --stage=refinement --beta=1.0 --checkpoint_name checkpoint-lfq_transplant_ImageNet_model_16_2_2_loss_1.0_0.4.pth.tar
													    
