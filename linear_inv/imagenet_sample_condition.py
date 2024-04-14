from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.guided_gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
import torchvision


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default="")
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--timestep', type=int, default=100)
    parser.add_argument('--eta', type=float, default=0.5)
    parser.add_argument('--scale', type=float, default=10)
    parser.add_argument('--method', type=str, default='mpgd_wo_proj')
    parser.add_argument('--save_dir', type=str, default='./outputs/imagenet/')
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    
    if args.timestep < 1000:
        diffusion_config["timestep_respacing"] = f"ddim{args.timestep}"
        diffusion_config["rescale_timesteps"] = True
    else:
        diffusion_config["timestep_respacing"] = f"1000"
        diffusion_config["rescale_timesteps"] = False
    
    diffusion_config["eta"] = args.eta
    task_config["conditioning"]["method"] = args.method
    task_config["conditioning"]["params"]["scale"] = args.scale
    task_config["data"]["name"] = "imagenet256"
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, resume = "../nonlinear/SD_style/models/ldm/cin256/model.ckpt", **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    dir_path = f"{diffusion_config['timestep_respacing']}_eta{args.eta}_scale{args.scale}"
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'], task_config['conditioning']['method'], dir_path)
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = torchvision.datasets.ImageFolder(root = "imagenet_root", transform = transform) # change your imagenet root here
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
        
    # Do Inference
    n_saved = 0
    c_set = []
    for i, (ref_img, c) in enumerate(loader):
        c = int(c)
        if c in c_set:
            continue
        c_set += [c]
        if n_saved >= 1000:
            break
        logger.info(f"Inference for image {i}, saved {len(c_set)}")
        fname = f'{i:05}_{c:05}.png'
        ref_img = ref_img.to(device)

        # Forward measurement model (Ax + n)
        y = operator.forward(ref_img)
        y_n = noiser(y)

        # Sampling
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path)

        plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))
        
        n_saved += 1

if __name__ == '__main__':
    main()
