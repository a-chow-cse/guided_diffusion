import os
def sample_inat():
    os.system("CUDA_VISIBLE_DEVICES=1,2 python classifier_sample.py --attention_resolutions 32,16,8 \
    --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True \
    --noise_schedule linear --num_channels 256 --num_heads 64 --num_res_blocks 2 \
    --resblock_updown True --use_fp16 True --use_scale_shift_norm True --classifier_scale 10.0 \
    --classifier_path ./models/pretrained_on_mimic_pair_3.pt --model_path models/256x256_diffusion_uncond.pt \
    --batch_size 2 --num_samples 4 --timestep_respacing 250")
def sample():
    os.system("CUDA_VISIBLE_DEVICES=1,2 python classifier_sample.py --attention_resolutions 32,16,8 \
    --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True \
    --noise_schedule linear --num_channels 256 --num_heads 64 --num_res_blocks 2 \
    --resblock_updown True --use_fp16 True --use_scale_shift_norm True --classifier_scale 10.0 \
    --classifier_path ./openai-2023-04-18-00-57-22-268248/model100000.pt --model_path models/256x256_diffusion_uncond.pt \
    --batch_size 2 --num_samples 4 --timestep_respacing 250")
def train():
    os.system("CUDA_VISIBLE_DEVICES=1,2 mpiexec -n N python classifier_train.py\
    --data_dir datasets/mimic_pair_3/train/ --val_data_dir datasets/mimic_pair_3/val/ \
    --iterations 300000 --anneal_lr True --batch_size 2 --lr 3e-4 --save_interval 10000 \
    --weight_decay 0.05 --image_size 256 --classifier_attention_resolutions 32,16,8 \
    --classifier_depth 2 --classifier_width 128 --classifier_pool attention \
    --classifier_resblock_updown True --classifier_use_scale_shift_norm True")
#sample()
sample_inat()
#train()