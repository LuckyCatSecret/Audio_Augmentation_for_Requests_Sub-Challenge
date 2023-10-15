#!/bin/bash
#SBATCH --job-name=aud
#SBATCH --partition=learnlab
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --mem=240GB
#SBATCH --signal=USR1@120
#SBATCH --constraint=volta32gb

#SBATCH --output=/checkpoint/%u/jobs/%A.out
#SBATCH --error=/checkpoint/%u/jobs/%A.err




blr=1e-3
ckpt=/YourPath/ckpt/pretrained.pth
audioset_train_json=/YourPath/HC-C/AudioMAE-main/dataset/hcc/hcc_complaint_train.json
audioset_eval_json=/YourPath/HC-C/AudioMAE-main/dataset/hcc/hcc_complaint_devel.json
audioset_label=/YourPath/HC-C/AudioMAE-main/dataset/hcc/hcc_complaint.csv
dataset=hcc

NCCL_BLOCKING_WAIT=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2223 \
aaa_only_finetune.py \
--log_dir ./hcc_complaint_finetune \
--output_dir ./hcc_complaint_finetune \
--model vit_base_patch16 \
--dataset $dataset \
--data_train $audioset_train_json \
--data_eval $audioset_eval_json \
--label_csv $audioset_label \
--finetune $ckpt \
--roll_mag_aug True \
--epochs 100 \
--blr $blr \
--batch_size 8 \
--warmup_epochs 4 \
--first_eval_ep 0 \
--dist_eval \
--mask_2d True \
--mask_t_prob 0.2 \
--mask_f_prob 0.2 \
--nb_classes 2
# --epochs 60
#--log_dir /checkpoint/berniehuang/mae/as_exp/$SLURM_JOB_ID \
#--output_dir /checkpoint/berniehuang/mae/as_exp/$SLURM_JOB_ID \