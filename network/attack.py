import torch 
import torch.nn as nn 
import torch.nn.functional as F
from functools import partial
import random


def KL(logit1,logit2,reverse=False):
    if reverse:
        logit1, logit2 = logit2, logit1
    p1 = logit1.softmax(1)
    logp1 = logit1.log_softmax(1)
    logp2 = logit2.log_softmax(1) 
    return (p1*(logp1-logp2)).sum(1)

def to_status(m, status):
    if hasattr(m, 'batch_type'):
        m.batch_type = status

to_clean_status = partial(to_status, status='clean')
to_adv_status = partial(to_status, status='adv')
to_mix_status = partial(to_status, status='mix')

def attack(fts_clean, label, classifier, final, nf_model, loss_flow, cfg, epsilon):
    classifier.apply(to_adv_status)

    # init noise
    if cfg['adv']['eps_rand_init']:
        noise = torch.empty_like(fts_clean).uniform_(-epsilon, epsilon)
    elif cfg['adv']['zero_init']:
        noise = torch.zeros_like(fts_clean)
    elif cfg['adv']['tiny_rand_init']:
        noise = (torch.rand_like(fts_clean) - 0.5) * 1e-6
    else:
        raise ValueError("Unknown noise init strategy in cfg['adv'].")

    noise.requires_grad_()

    # 关键：NF/Flow 这段强制 fp32，避免 autocast 溢出
    with torch.cuda.amp.autocast(enabled=False):
        fts_pt = (fts_clean.float() + noise.float())
        fts_final = classifier(fts_pt).float()  # [B, C, h, w]

        label = label.unsqueeze(1).float()
        label_nf = F.interpolate(label, size=fts_final.shape[2:], mode="nearest").squeeze(1).long()

        label_flat = label_nf.view(-1)
        valid_mask = (label_flat != 255)
        if valid_mask.sum() == 0:
            classifier.apply(to_mix_status)
            return torch.zeros_like(fts_clean)

        fts_flat = fts_final.permute(0, 2, 3, 1).reshape(-1, fts_final.shape[1])
        fts_valid = fts_flat[valid_mask].contiguous()
        y_valid = label_flat[valid_mask].contiguous()

        # NF forward
        output_z, ljd = nf_model(fts_valid)
        # 保险：限制 log_det（避免极端值把 ll 拉爆）
        ljd = torch.nan_to_num(ljd, nan=0.0, posinf=0.0, neginf=0.0).clamp(-100.0, 100.0)

        nf_loss, _, _, _ = loss_flow(output_z, sldj=ljd, y=y_valid)

        # 非有限直接返回 0 扰动，防止污染
        if not torch.isfinite(nf_loss):
            classifier.apply(to_mix_status)
            return torch.zeros_like(fts_clean)

        grad = torch.autograd.grad(nf_loss, noise, retain_graph=False, create_graph=False)[0]
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        den = torch.norm(grad, p=2, dim=1, keepdim=True).clamp_min(1e-6)
        noise_out = epsilon * (grad / den)
        noise_out = torch.nan_to_num(noise_out, nan=0.0, posinf=0.0, neginf=0.0)

    classifier.apply(to_mix_status)
    return noise_out


# def attack(fts_clean, label, classifier, final, nf_model, loss_flow, cfg, epsilon):
#     classifier.apply(to_adv_status)
#
#     # 初始化 noise
#     if cfg['adv']['eps_rand_init']:
#         noise = torch.empty_like(fts_clean).uniform_(-epsilon, epsilon)
#     elif cfg['adv']['zero_init']:
#         noise = torch.zeros_like(fts_clean)
#     elif cfg['adv']['tiny_rand_init']:
#         noise = (torch.rand_like(fts_clean) - 0.5) * 1e-6
#     else:
#         raise ValueError("Unknown noise init strategy in cfg['adv'].")
#
#     noise.requires_grad_()
#     fts_clean.requires_grad_()
#
#     fts_pt = fts_clean + noise
#     fts_final = classifier(fts_pt)  # [B, C, h, w]
#
#     # 处理 label（伪标签）
#     label = label.unsqueeze(1).float()  # [B, 1, H, W]
#     label_nf = F.interpolate(label, size=fts_final.shape[2:], mode="nearest").squeeze(1).long()  # [B, h, w]
#
#     # 展平
#     label_flat = label_nf.view(-1)
#     fts_final_flat = fts_final.permute(0, 2, 3, 1).reshape(-1, fts_final.shape[1])  # [B*h*w, C]
#
#     final_loss = 0
#     # 只根据 label 掩码（无需对 fts 再掩码）
#     valid_mask = (label_flat != 255)
#     if valid_mask.sum() == 0:
#         # print("valid_mask",valid_mask.sum())
#         classifier.apply(to_mix_status)
#         return torch.zeros_like(fts_clean)
#
#     fts_final_flat_valid = fts_final_flat[valid_mask]
#     label_flat_valid = label_flat[valid_mask]
#
#
#     # print(f"DEBUG: fts_final_flat shape = {fts_final_flat.shape}")
#     # print(f"DEBUG: label_flat shape = {label_flat.shape}")
#     # print(f"DEBUG: valid_mask shape = {valid_mask.shape}")
#     # print(f"DEBUG: label_flat_valid shape = {label_flat_valid.shape}")
#     # print(f"DEBUG: fts_final_flat_valid count = {fts_final_flat_valid.shape[0]}")
#
#     output_z, ljd = nf_model(fts_final_flat_valid)
#     nf_loss, _, _, _ = loss_flow(output_z, sldj=ljd)
#
#     final_loss += nf_loss
#
#     # 计算扰动
#     grad = torch.autograd.grad(final_loss, noise, retain_graph=False, create_graph=False)[0]
#     noise = epsilon * F.normalize(grad.detach(), dim=1, p=2)
#
#     classifier.apply(to_mix_status)
#     return noise
#




