python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/super_resolution_config.yaml \
    --timestep=100 \
    --scale=5 \
    --method="mpgd_ae" \


python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/super_resolution_config.yaml \
    --timestep=50 \
    --scale=10 \
    --method="mpgd_ae" \


python ffhq_sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/mgpd_diffusion_config.yaml \
    --task_config=configs/super_resolution_config.yaml \
    --timestep=20 \
    --scale=20 \
    --method="mpgd_ae" \

