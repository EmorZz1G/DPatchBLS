import sys
from ..layer.patch_bls import KernelBLSBlock, BasicBLSBlock
import torch.nn as nn
from bls_torch.utils.activation import ACTIVATION
import torch
from ..patch_utils import get_patch_num
sys.path.append('..')


class TwoPatchBLS_Single(nn.Module):
    def __init__(self, seq_len, kernel_patch_size, basic_patch_size, kernel_dim, sigma,
                 n_feature_dim,
                 n_feature_group,
                 n_enhancement_dim,
                 n_enhancement_group,
                 n_feature_cascade_layer,
                 n_enhancement_cascade_layer,
                 enhancement_s=0.8,
                 output_reg=2 ** -1,
                 feature_activation=nn.Sigmoid(),
                 enhancement_activation=nn.Sigmoid(),
                 feature_cascade_activation=ACTIVATION.none,
                 enhancement_cascade_activation=ACTIVATION.none,
                 use_positional=False,
                 max_len=0,
                 auto_encoder=True,
                 conv_feature_builder=False,
                 device=None
                 ) -> None:
        super().__init__()

        kernel_patch_num = get_patch_num(seq_len, kernel_patch_size)
        basic_patch_num = get_patch_num(seq_len, basic_patch_size)

        self.kernel_net = KernelBLSBlock(kernel_patch_num, kernel_patch_size, kernel_dim, sigma,
                                         n_feature_dim,
                                         n_feature_group,
                                         n_enhancement_dim,
                                         n_enhancement_group,
                                         n_feature_cascade_layer,
                                         n_enhancement_cascade_layer,
                                         enhancement_s,
                                         output_reg,
                                         feature_activation,
                                         enhancement_activation,
                                         feature_cascade_activation,
                                         enhancement_cascade_activation,
                                         use_positional,
                                         max_len,
                                         auto_encoder,
                                         conv_feature_builder,
                                         device)

        self.basic_net = BasicBLSBlock(basic_patch_num, basic_patch_size,
                                       n_feature_dim,
                                       n_feature_group,
                                       n_enhancement_dim,
                                       n_enhancement_group,
                                       n_feature_cascade_layer,
                                       n_enhancement_cascade_layer,
                                       enhancement_s,
                                       output_reg,
                                       feature_activation,
                                       enhancement_activation,
                                       feature_cascade_activation,
                                       enhancement_cascade_activation,
                                       use_positional,
                                       max_len,
                                       auto_encoder,
                                       conv_feature_builder,
                                       device)

    def forward(self, x, y):
        x1 = x2 = x
        k_res = self.kernel_net(x1, x1)
        b_res = self.basic_net(x2, x2)
        k_res = torch.sigmoid(k_res)
        b_res = torch.sigmoid(b_res)
        return k_res, b_res

    def decision_function(self, x, verbose=False):
        x1 = x2 = x
        s1 = self.kernel_net.decision_function(x1)
        s2 = self.basic_net.decision_function(x2)
        s1_s = torch.sigmoid(s1)
        s2_s = torch.sigmoid(s2)
        # Apply the KL Divergence to compute the dissimilarity between the two representations.
        s = 0.5 * (s1_s * (torch.log(s1_s + 1e-7) - torch.log(s2_s + 1e-7))) + 0.5 * (s2_s * (torch.log(s2_s + 1e-7) - torch.log(s1_s + 1e-7)))
        if verbose:
            print('Kernel-BLS', torch.mean(s1).item())
            print('Basic-BLS', torch.mean(s2).item())
            print('TwoPatch-BLS', torch.mean(s).item())
        return s1, s2, s
