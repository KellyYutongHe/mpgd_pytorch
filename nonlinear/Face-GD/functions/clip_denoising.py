import torch
from tqdm import tqdm
import torchvision.utils as tvu
import torchvision
import os
import pickle

from .clip.base_clip import CLIPEncoder

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def wo_proj_clip_ddim_diffusion(x, seq, model, b, rho_scale=1.0, prompt=None, stop=100, repeat=5, repeat_start=700, repeat_end=400):
    clip_encoder = CLIPEncoder().cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    
    og_rho_scale = rho_scale
    og_repeat = repeat

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')

        if repeat_start >= i >= repeat_end:
            repeat = og_repeat
            rho_scale = og_rho_scale
        else:
            repeat = 1
            rho_scale = 1
        
        for idx in range(repeat):
            et = model(xt, t)

            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_t = x0_t.detach()
            x0_t.requires_grad = True
            
            # get guided gradient
            residual = clip_encoder.get_residual(x0_t, prompt)
            norm = torch.linalg.norm(residual)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x0_t)[0]
            
            x0_t = x0_t.detach()

            # hard coded eta = 1, TODO: make eta flexible
            c1 = at_next.sqrt() * (1 - at / at_next) / (1 - at)
            c2 = (at / at_next).sqrt() * (1 - at_next) / (1 - at)
            c3 = (1 - at_next) * (1 - at / at_next) / (1 - at)
            c3 = (c3.log() * 0.5).exp()
            
            l1 = ((et * et).mean().sqrt() * (1 - at).sqrt() / at.sqrt() * c1).item()
            l2 = l1 * rho_scale
            rho = l2 / (norm_grad * norm_grad).mean().sqrt().item()
            rho = rho / at_next.sqrt()
            
            if not i <= stop:
                x0_t -= rho * norm_grad
            xt_next = c1 * x0_t + c2 * xt + c3 * torch.randn_like(x0_t)
            
            xt_next = xt_next.detach()
            
            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))

            if idx + 1 < repeat:
                bt = at / at_next
                xt = bt.sqrt() * xt_next + (1 - bt).sqrt() * torch.randn_like(xt_next)
        
    # return x0_preds, xs
    return [xs[-1]], [x0_preds[-1]]


def ae_clip_ddim_diffusion(x, seq, model, b, rho_scale=1.0, prompt=None, stop=100, repeat=5, repeat_start=700, repeat_end=400, ldm_start=500, ldm_end=300, ldm_model=None):
    clip_encoder = CLIPEncoder().cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    
    og_rho_scale = rho_scale
    og_repeat = repeat

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')

        if repeat_start >= i >= repeat_end:
            repeat = og_repeat
            rho_scale = og_rho_scale
        else:
            repeat = 1
            rho_scale = 1
        
        for idx in range(repeat):
            et = model(xt, t)

            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_t = x0_t.detach()
            x0_t.requires_grad = True
            
            # get guided gradient
            if ldm_start > i > ldm_end:
                E_x0_t = ldm_model.encode_first_stage(x0_t)
                D_x0_t = ldm_model.decode_first_stage(E_x0_t)
                residual = clip_encoder.get_residual(D_x0_t, prompt)
            else:
                residual = clip_encoder.get_residual(x0_t, prompt)
            norm = torch.linalg.norm(residual)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x0_t)[0]
            
            x0_t = x0_t.detach()

            # hard coded eta = 1, TODO: make eta flexible
            c1 = at_next.sqrt() * (1 - at / at_next) / (1 - at)
            c2 = (at / at_next).sqrt() * (1 - at_next) / (1 - at)
            c3 = (1 - at_next) * (1 - at / at_next) / (1 - at)
            c3 = (c3.log() * 0.5).exp()
            
            l1 = ((et * et).mean().sqrt() * (1 - at).sqrt() / at.sqrt() * c1).item()
            l2 = l1 * rho_scale
            rho = l2 / (norm_grad * norm_grad).mean().sqrt().item()
            rho = rho / at_next.sqrt()
            
            if not i <= stop:
                x0_t -= rho * norm_grad
            xt_next = c1 * x0_t + c2 * xt + c3 * torch.randn_like(x0_t)
            
            xt_next = xt_next.detach()
            
            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))

            if idx + 1 < repeat:
                bt = at / at_next
                xt = bt.sqrt() * xt_next + (1 - bt).sqrt() * torch.randn_like(xt_next)
        
    # return x0_preds, xs
    return [xs[-1]], [x0_preds[-1]]

def z_clip_ddim_diffusion(x, seq, model, b, rho_scale=1.0, prompt=None, stop=100, repeat=5, repeat_start=700, repeat_end=400, ldm_start=500, ldm_end=300, ldm_model=None):
    clip_encoder = CLIPEncoder().cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    
    og_rho_scale = rho_scale
    og_repeat = repeat

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')

        if repeat_start >= i >= repeat_end:
            repeat = og_repeat
            rho_scale = og_rho_scale
        else:
            repeat = 1
            rho_scale = 1
        
        for idx in range(repeat):
            et = model(xt, t)

            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_t = x0_t.detach()
            x0_t.requires_grad = True
            
            # hard coded eta = 1, TODO: make eta flexible
            c1 = at_next.sqrt() * (1 - at / at_next) / (1 - at)
            c2 = (at / at_next).sqrt() * (1 - at_next) / (1 - at)
            c3 = (1 - at_next) * (1 - at / at_next) / (1 - at)
            c3 = (c3.log() * 0.5).exp()
            
            if i > stop and (ldm_start > i > ldm_end):
                E_x0_t = ldm_model.encode_first_stage(x0_t)
                E_x0_t = E_x0_t.detach()
                E_x0_t.requires_grad = True
                D_x0_t = ldm_model.decode_first_stage(E_x0_t)
                diff = x0_t.detach() - D_x0_t.detach()
                residual = clip_encoder.get_residual(D_x0_t, prompt)
                norm = torch.linalg.norm(residual)
                norm_grad = torch.autograd.grad(outputs=norm, inputs=E_x0_t)[0]
            
                l1 = ((et * et).mean().sqrt() * (1 - at).sqrt() / at.sqrt() * c1).item()
                l2 = l1 * rho_scale
                rho = l2 / (norm_grad * norm_grad).mean().sqrt().item()
                rho = rho / at_next.sqrt()
                
                E_x0_t = E_x0_t.detach()
                E_x0_t -= rho*norm_grad
                D_x0_t = ldm_model.decode_first_stage(E_x0_t)
                x0_t = D_x0_t.detach() + diff
            elif i > stop:
                x0_t.requires_grad = True
                residual = clip_encoder.get_residual(x0_t, prompt)
                norm = torch.linalg.norm(residual)
                norm_grad = torch.autograd.grad(outputs=norm, inputs=x0_t)[0]
                
                l1 = ((et * et).mean().sqrt() * (1 - at).sqrt() / at.sqrt() * c1).item()
                l2 = l1 * rho_scale
                rho = l2 / (norm_grad * norm_grad).mean().sqrt().item()
                rho = rho / at_next.sqrt()
                
                x0_t = x0_t.detach()
                x0_t -= rho * norm_grad
            
            xt_next = c1 * x0_t + c2 * xt + c3 * torch.randn_like(x0_t)
            
            xt_next = xt_next.detach()
            
            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))

            if idx + 1 < repeat:
                bt = at / at_next
                xt = bt.sqrt() * xt_next + (1 - bt).sqrt() * torch.randn_like(xt_next)

    # return x0_preds, xs
    return [xs[-1]], [x0_preds[-1]]


def sample_ddim(x, seq, model, b, repeat=5, repeat_start=700, repeat_end=400):
    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    
    og_repeat = repeat

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        
        if repeat_start >= i >= repeat_end:
            repeat = og_repeat
        else:
            repeat = 1
        
        for idx in range(repeat):
        
            et = model(xt, t)

            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = at_next.sqrt() * (1 - at / at_next) / (1 - at)
            c2 = (at / at_next).sqrt() * (1 - at_next) / (1 - at)
            c3 = (1 - at_next) * (1 - at / at_next) / (1 - at)
            c3 = (c3.log() * 0.5).exp()
            xt_next = c1 * x0_t + c2 * xt + c3 * torch.randn_like(x0_t)

            x0_t = x0_t.detach()
            xt_next = xt_next.detach()

            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))
            
            if idx + 1 < repeat:
                bt = at / at_next
                xt = bt.sqrt() * xt_next + (1 - bt).sqrt() * torch.randn_like(xt_next)

    return [xs[-1]], [x0_preds[-1]]
