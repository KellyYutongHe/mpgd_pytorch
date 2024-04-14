from abc import ABC, abstractmethod
import torch
import numpy as np
import os

import sys
from omegaconf import OmegaConf
from .util import instantiate_from_config
import glob

__CONDITIONING_METHOD__ = {}

def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    # model.eval()
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

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t
    
@register_conditioning_method(name='mpgd_wo_proj')
class WoProjSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_0_hat, measurement, at, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_0_hat, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_0_hat = x_0_hat.detach()
        x_0_hat -= norm_grad * self.scale / at.sqrt()
        return x_0_hat, norm

@register_conditioning_method(name='mpgd_ae')
class AESampling(ConditioningMethod):
    def __init__(self, operator, noiser, resume, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        ckpt = None
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

        self.ldm_model, _ = load_model(config, ckpt, gpu, eval_mode)
        
    def conditioning(self, x_0_hat, measurement, at, **kwargs):
        t = kwargs.get('t', 0.5)
        if 0.5 > t > 0.3:
            E_x0_t = self.ldm_model.encode_first_stage(x_0_hat)
            D_x0_t = self.ldm_model.decode_first_stage(E_x0_t)
            norm_grad, norm = self.grad_and_value(x_prev=x_0_hat, x_0_hat=D_x0_t, measurement=measurement, **kwargs)
            x_0_hat = x_0_hat.detach()
            x_0_hat -= norm_grad * self.scale*0.0075 / at.sqrt()
        else:
            norm_grad, norm = self.grad_and_value(x_prev=x_0_hat, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
            x_0_hat = x_0_hat.detach()
            x_0_hat -= norm_grad * self.scale / at.sqrt()
        return x_0_hat, norm

@register_conditioning_method(name='mpgd_z')
class LatentSampling(ConditioningMethod):
    def __init__(self, operator, noiser, resume, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        ckpt = None
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

        self.ldm_model, _ = load_model(config, ckpt, gpu, eval_mode)
        

    def conditioning(self, x_0_hat, measurement, at, **kwargs):
        t = kwargs.get('t', 0.5)
        if 0.5 > t > 0.3:
            E_x0_t = self.ldm_model.encode_first_stage(x_0_hat)
            E_x0_t = E_x0_t.detach()
            E_x0_t.requires_grad = True
            D_x0_t = self.ldm_model.decode_first_stage(E_x0_t)
            diff = x_0_hat.detach() - D_x0_t.detach()
            norm_grad, norm = self.grad_and_value(x_prev=E_x0_t, x_0_hat=D_x0_t, measurement=measurement, **kwargs)
            E_x0_t = E_x0_t.detach()
            E_x0_t -= self.scale*0.01*norm_grad / at[0,0,0,0].sqrt()
            D_x0_t = self.ldm_model.decode_first_stage(E_x0_t)
            x_0_hat = D_x0_t.detach() + diff
            x_0_hat = x_0_hat.detach()
        else:
            norm_grad, norm = self.grad_and_value(x_prev=x_0_hat, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
            x_0_hat = x_0_hat.detach()
            x_0_hat -= norm_grad * self.scale / at.sqrt()
        return x_0_hat, norm