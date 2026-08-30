#!/bin/bash
#SBATCH --job-name=fsq_ffhq_refinement
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --nodes=1
#SBATCH --mem=80gb
#SBATCH --cpus-per-task 5
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --output /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/slurm/Refinement/FFHQ/fsq_ffhq_refinement.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/Rate-Distortion-Perspective/VAR-Transplant/slurm/Refinement/FFHQ/fsq_ffhq_refinement.err

module load gcc opencv/4.8.1
source /home/sunset/environment/VQ-Tokenizer/bin/activate
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=25582 train_refinement.py --VQ=fsq --dataset_name=FFHQ --global_batch_size=64  --L=4 --project_dim=8 --stage=refinement --beta=1.0 --checkpoint_name=checkpoint-fsq_transplant_FFHQ_model_8_4_loss_1.0_0.4.pth.tar
