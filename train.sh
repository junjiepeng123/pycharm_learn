python train.py \
  --train_lr_dir ./Datasets/VSR/REDS/train/bicubic \
  --train_hr_dir ./Datasets/VSR/REDS/train/GT \
  --val_lr_dir ./Datasets/VSR/REDS/val/bicubic \
  --val_hr_dir ./Datasets/VSR/REDS/val/GT \
  --batch_size 4 \
  --num_epochs 100 \
  --model SimpleVSR \
  --exp_name reds_vsr \
  --gpu_id 0