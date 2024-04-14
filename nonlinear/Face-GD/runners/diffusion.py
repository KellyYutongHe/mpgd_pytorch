import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from functions.ckpt_util import get_ckpt_path, download
from functions.clip_denoising import wo_proj_clip_ddim_diffusion, sample_ddim, ae_clip_ddim_diffusion, z_clip_ddim_diffusion
from functions.faceid_denoising import wo_proj_faceid_ddim_diffusion, ae_faceid_ddim_diffusion, z_faceid_ddim_diffusion
import torchvision.utils as tvu
from torchvision.datasets import ImageFolder, CelebA
from torchvision import transforms, utils

from guided_diffusion.unet import UNetModel
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random

import sys
sys.path.append('..')
from functions.arcface.model import IDLoss

import sys
from omegaconf import OmegaConf
from .ldm.ldm.util import instantiate_from_config

def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model

def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])
    return model, global_step

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, device=None):
        self.args = args
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = "fixedsmall"
        betas = get_beta_schedule(
            beta_schedule="linear",
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=1000,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self, mode):
        model_f = None

        if self.args.model_type == "face":
            # get face model
            celeba_dict = {
                'type': "simple",
                'in_channels': 3,
                'out_ch': 3,
                'ch': 128,
                'ch_mult': [1, 1, 2, 2, 4, 4],
                'num_res_blocks': 2,
                'attn_resolutions': [16, ],
                'dropout': 0.0,
                'var_type': 'fixedsmall',
                'ema_rate': 0.999,
                'ema': True,
                'resamp_with_conv': True,
                "image_size": 256, 
                "resamp_with_conv": True,
                "num_diffusion_timesteps": 1000,
            }
            model_f = Model(celeba_dict)
            ckpt = os.path.join(self.args.exp, "models/celeba_hq.ckpt")
            states = torch.load(ckpt, map_location=self.device)
            if type(states) == list:
                states_old = states[0]
                states = dict()
                for k, v in states.items():
                    states[k[7:]] = v
            else:
                model_f.load_state_dict(states)
            model_f.to(self.device)
            model_f = torch.nn.DataParallel(model_f)
            model = model_f


        if mode == "faceid":
            self.sample_faceid(model, mode)
        elif mode == "face_clip":
            self.sample_face_clip(model, mode)
    
    
    def sample_face_clip(self, model, mode):
        args = self.args
        prompt = args.prompt
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        n_saved = 0
        
        ckpt = None
        resume = args.resume

        if not os.path.exists(resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(resume):
            try:
                resumedir = '/'.join(resume.split('/')[:-1])
            except ValueError:
                paths = resume.split("/")
                idx = -2
                resumedir = "/".join(paths[:idx])
            ckpt = resume
        else:
            assert os.path.isdir(resume), f"{resume} is not a directory"
            resumedir = resume.rstrip("/")
            ckpt = os.path.join(resumedir, "model.ckpt")

        base_configs = sorted(glob.glob(os.path.join(resumedir, "config.yaml")))

        configs = [OmegaConf.load(cfg) for cfg in base_configs]
        cli = OmegaConf.from_dotlist([])
        config = OmegaConf.merge(*configs, cli)

        gpu = True
        eval_mode = True

        ldm_model, _ = load_model(config, ckpt, gpu, eval_mode)
        
        for n in range(self.args.batch_size):
            x = torch.randn(
                1,
                3,
                256,
                256,
                device=self.device,
            )
            if args.uncond:
                x, _ = sample_ddim(x, seq, model, self.betas, repeat=args.repeat, repeat_start=args.repeat_start, repeat_end=args.repeat_end)
            elif args.method == "ae":
                x, _ = ae_clip_ddim_diffusion(x, seq, model, self.betas, prompt=prompt, rho_scale=args.rho_scale, stop=args.stop, repeat=args.repeat, repeat_start=args.repeat_start, repeat_end=args.repeat_end, ldm_start=args.ldm_start, ldm_end=args.ldm_end, ldm_model=ldm_model)
            elif args.method == "z":
                x, _ = z_clip_ddim_diffusion(x, seq, model, self.betas, prompt=prompt, rho_scale=args.rho_scale, stop=args.stop, repeat=args.repeat, repeat_start=args.repeat_start, repeat_end=args.repeat_end, ldm_start=args.ldm_start, ldm_end=args.ldm_end, ldm_model=ldm_model)
            else:
                x, _ = wo_proj_clip_ddim_diffusion(x, seq, model, self.betas, prompt=prompt, rho_scale=args.rho_scale, stop=args.stop, repeat=args.repeat, repeat_start=args.repeat_start, repeat_end=args.repeat_end)
            x = [((y + 1.0) / 2.0).clamp(0.0, 1.0) for y in x]
            for i in range(len(x)):
                for j in range(x[i].size(0)):
                    tvu.save_image(
                        x[i][j], os.path.join(self.args.image_folder, f"{prompt}_{n_saved:04}.png")
                    )
                    n_saved += 1

    def sample_faceid(self, model, mode):
        args = self.args
        ref_path = args.ref_path
        idloss = IDLoss().cuda()
        
        ckpt = None
        resume = args.resume

        if not os.path.exists(resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(resume):
            try:
                resumedir = '/'.join(resume.split('/')[:-1])
            except ValueError:
                paths = resume.split("/")
                idx = -2
                resumedir = "/".join(paths[:idx])
            ckpt = resume
        else:
            assert os.path.isdir(resume), f"{resume} is not a directory"
            resumedir = resume.rstrip("/")
            ckpt = os.path.join(resumedir, "model.ckpt")

        base_configs = sorted(glob.glob(os.path.join(resumedir, "config.yaml")))

        configs = [OmegaConf.load(cfg) for cfg in base_configs]
        cli = OmegaConf.from_dotlist([])
        config = OmegaConf.merge(*configs, cli)

        gpu = True
        eval_mode = True

        ldm_model, _ = load_model(config, ckpt, gpu, eval_mode)
        
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        n_saved = 0
        
        idloss.calc_ref_feat(ref_path)
        for index in range(self.args.batch_size):
            x = torch.randn(
                1,
                3,
                256,
                256,
                device=self.device,
            )
            if args.uncond:
                x, _ = sample_ddim(x, seq, model, self.betas, repeat=args.repeat, repeat_start=args.repeat_start, repeat_end=args.repeat_end)
            elif args.method == "ae":
                x, _ = ae_faceid_ddim_diffusion(x, seq, model, self.betas, eta=args.eta, idloss=idloss, rho_scale=args.rho_scale, stop=args.stop, repeat=args.repeat, repeat_start=args.repeat_start, repeat_end=args.repeat_end, ldm_start=args.ldm_start, ldm_end=args.ldm_end, ldm_model=ldm_model)
            elif args.method == "z":
                x, _ = z_faceid_ddim_diffusion(x, seq, model, self.betas, eta=args.eta, idloss=idloss, rho_scale=args.rho_scale, stop=args.stop, repeat=args.repeat, repeat_start=args.repeat_start, repeat_end=args.repeat_end, ldm_start=args.ldm_start, ldm_end=args.ldm_end, ldm_model=ldm_model)
            else:
                x, _ = wo_proj_faceid_ddim_diffusion(x, seq, model, self.betas, eta=args.eta, idloss=idloss, rho_scale=args.rho_scale, stop=args.stop, repeat=args.repeat, repeat_start=args.repeat_start, repeat_end=args.repeat_end)
            x = [((y + 1.0) / 2.0).clamp(0.0, 1.0) for y in x]
            for i in range(len(x)):
                for j in range(x[i].size(0)):
                    tvu.save_image(
                        x[i][j], os.path.join(self.args.image_folder, f"{n_saved:04}_{index}.png")
                    )
                    n_saved += 1