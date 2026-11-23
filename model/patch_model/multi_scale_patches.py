import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from model.patch_model.patch_bls_single import TwoPatchBLS_Single
from model.trainer import trainer, test


@torch.no_grad()
def multi_scale_patches_model(patches_list, train_x, test_x, pre_trained_weight, validation_size, data_name, **bls_param):
    
    seq_len, n_vars = train_x.shape

    n_train = int((1 - validation_size) * seq_len)
    train_x_split = train_x[:n_train]
    valid_x_split = train_x[n_train:]

    KB = None
    B = None
    P = None
    TR_K = None
    TR_B = None
    VA_K = None
    VA_B = None
    T_tr = 0
    T_te = 0

    save_dir = f"./weights/{data_name}"
    os.makedirs(save_dir, exist_ok=True)

    if not pre_trained_weight:
        if os.listdir(save_dir):
            for file in os.listdir(save_dir):
                os.remove(os.path.join(save_dir, file))

    for patch_size in patches_list:

        print(f'\nProcessing model with patch size {patch_size}:')
        kbs, bs, ps, tr_ks, tr_bs, va_ks, va_bs = [], [], [], [], [], [], []
        t_tr_single = t_te_single = 0
        variables = tqdm(range(n_vars), file=sys.stdout)
        for variable in variables:
            train_x_variable = train_x_split[:, variable]
            valid_x_variable = valid_x_split[:, variable]
            test_x_variable = test_x[:, variable]
            weight_path = f"./weights/{data_name}/variable_{variable + 1}_patch_size_{patch_size}.pt"
            if not pre_trained_weight:
                model = TwoPatchBLS_Single(seq_len, kernel_patch_size=patch_size, basic_patch_size=patch_size, **bls_param)
                kb, b, p, t_tr, t_te, tr_k, tr_b, va_k, va_b = trainer(model, train_x_variable, valid_x_variable, test_x_variable, device=bls_param['device'])
                torch.save(deepcopy(model), weight_path)
                t_tr_single += t_tr
                t_te_single += t_te
                tr_ks.append(tr_k)
                tr_bs.append(tr_b)
                va_ks.append(va_k)
                va_bs.append(va_b)
            else:
                model = torch.load(weight_path, map_location=torch.device(device=bls_param['device']))
                kb, b, p, t_te = test(model, test_x_variable, device=bls_param['device'])
                t_te_single += t_te
            kbs.append(kb)
            bs.append(b)
            ps.append(p)

        kbs_stack = np.stack(kbs)
        bs_stack = np.stack(bs)
        ps_stack = np.stack(ps)

        tr_ks_stack = np.stack(tr_ks) if tr_ks else None
        tr_bs_stack = np.stack(tr_bs) if tr_bs else None
        va_ks_stack = np.stack(va_ks) if va_ks else None
        va_bs_stack = np.stack(va_bs) if va_bs else None

        if KB is None:
            KB = np.mean(kbs_stack, axis=0)
            B = np.mean(bs_stack, axis=0)
            P = np.mean(ps_stack, axis=0)
            if tr_ks_stack is not None:
                TR_K = tr_ks_stack
                TR_B = tr_bs_stack
                VA_K = va_ks_stack
                VA_B = va_bs_stack
        else:
            KB += np.mean(kbs_stack, axis=0)
            B += np.mean(bs_stack, axis=0)
            P += np.mean(ps_stack, axis=0)
            if tr_ks_stack is not None:
                TR_K += tr_ks_stack
                TR_B += tr_bs_stack
                VA_K += va_ks_stack
                VA_B += va_bs_stack

        T_tr += t_tr_single
        T_te += t_te_single

        print('The model with the patch size of %d has the training duration of: %f s' % (patch_size, t_tr_single))
        print('The model with the patch size of %d has the testing duration of: %f s' % (patch_size, t_te_single))

    KB = KB / len(patches_list)
    B = B / len(patches_list)
    P = P / len(patches_list)
    
    if TR_K is not None:
        TR_K = TR_K / len(patches_list)
        TR_B = TR_B / len(patches_list)
        VA_K = VA_K / len(patches_list)
        VA_B = VA_B / len(patches_list)
        TR_K_tensor = torch.tensor(TR_K)
        TR_B_tensor = torch.tensor(TR_B)
        VA_K_tensor = torch.tensor(VA_K)
        VA_B_tensor = torch.tensor(VA_B)
        tr_kl = 0.5 * (TR_K_tensor * (torch.log(TR_K_tensor + 1e-7) - torch.log(TR_B_tensor + 1e-7))) + \
                0.5 * (TR_B_tensor * (torch.log(TR_B_tensor + 1e-7) - torch.log(TR_K_tensor + 1e-7)))
        va_kl = 0.5 * (VA_K_tensor * (torch.log(VA_K_tensor + 1e-7) - torch.log(VA_B_tensor + 1e-7))) + \
                0.5 * (VA_B_tensor * (torch.log(VA_B_tensor + 1e-7) - torch.log(VA_K_tensor + 1e-7)))
        tr_kl_mean = torch.mean(tr_kl, dim=0)
        va_kl_mean = torch.mean(va_kl, dim=0)
        TR_LOSS = float(torch.sum(tr_kl_mean))
        VA_LOSS = float(torch.sum(va_kl_mean))
    else:
        TR_LOSS = VA_LOSS = 0.0

    return P, T_tr, T_te, TR_LOSS, VA_LOSS

