accelerate launch  --multi_gpu --num_processes=8 --gpu_ids="0,1,2,3,4,5,6,7" train_cam.py \
--output_dir="./outputs_train" \
--tracker_project_name="tf" \
--train_data_csv="data/example_train_data.csv" \
--val_data_csv="data/example_val_data.csv" \
--learning_rate=5e-5 \
--gradient_accumulation_steps=1 \
--lr_warmup_steps=1000 \
--scale_lr \
--train_batch_size=1 \
--n_sample_frames=8 \
--sample_frame_stride=5 \
--h=256 \
--w=256 \
--proportion_empty_prompts=0.05 \
--max_train_steps=100001 \
--validation_interval=10000 \
--checkpointing_interval=10000 \
--mixed_precision='no' \
--allow_tf32 \
--val_sanity_check \
--gradient_checkpointing \
# --resume_from_checkpoint="./ckpt/xxx.pth"

