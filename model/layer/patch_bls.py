import torch
import torch.nn as nn
from bls_torch.layers import Feature, Enhancement, Output
from bls_torch.layers import CascadeKernelFeatureGroup, KernelFeatureGroup
from bls_torch.layers.outdim import LatentOutDim
from bls_torch.layers import (
    FeatureGroup,
    EnhancementGroup,
    CascadeFeatureGroup,
    CascadeEnhancementGroup,
)
from bls_torch.utils.activation import ACTIVATION
from model.layer.emb import PositionalEmbedding
import sys
from model.patch_utils import patch2series, patch_fix
sys.path.append('..')


class AutoEncoder:
    """
    only need to implement `transform`
    """
    def fit(self, x, x_true=None):
        assert self.auto_encoder, "auto encoder should be true"
        self.train()
        if x_true is not None:
            self(x, x_true)
            self.eval()
        else:
            self(x, x)
            self.eval()

    def transform(self, x):
        """
        only use to validate state, should be implemented manually.
        :param x:
        :return:
        """
        assert self.auto_encoder, "auto encoder should be true"
        assert self.training is False, "model is not in eval state"

    def fit_transform(self, x, x_true=None):
        assert self.auto_encoder, "auto encoder should be true"
        self.train()
        if x_true is not None:
            self(x, x_true)
            self.eval()
            return self.transform(x)
        else:
            self(x, x)
            self.eval()
            return self.transform(x)


    __constants__ = [
        "in_features",
        "n1_feature_node_dim",
        "n2_feature_mapping_group",
        "n3_enhancement_dim",
        "out_features",
        "out_layer_reg",
    ]

    def fit_transform(self, x):
        return super().fit_transform(x)

    def __init__(
            self,
            in_features,
            out_features,
            n1_feature_node_dim,
            n2_feature_mapping_group,
            n3_enhancement_dim,
            n4_enhancement_mapping_group=2,
            feature_activation=nn.Sigmoid(),
            enhancement_activation=nn.Sigmoid(),
            enhancement_shrink=0.8,
            out_layer_reg=2 ** -4,
            auto_encoder=False,
    ):
        """
        :param in_features: must match input dim
        :param n1_feature_node_dim: feature node out dim
        :param n2_feature_mapping_group: the num of feature mapping group
        :param n3_enhancement_dim: enhancement layer out dim (all groups)
        :param n4_enhancement_mapping_group:the num of enhancement mapping group
        :param out_features: out dim, eq to num of class or in dim
        :param enhancement_shrink: shrink coefficient
        :param out_layer_reg: regularization coefficient
        """
        super(AutoEncoder, self).__init__()
        self.auto_encoder = auto_encoder
        if self.auto_encoder:
            assert in_features == out_features, "in and out must be equal"
        self.feature_activation = feature_activation
        self.enhancement_activation = enhancement_activation
        self.feature_layers = nn.ModuleList()
        for i in range(n2_feature_mapping_group):
            self.feature_layers.add_module(
                "Z{}".format(i + 1), Feature(in_features, n1_feature_node_dim)
            )
        self.enhancement_layers = nn.ModuleList()
        for i in range(n4_enhancement_mapping_group):
            el = Enhancement(
                n1_feature_node_dim * n2_feature_mapping_group,
                n3_enhancement_dim,
                s=enhancement_shrink,
            )
            self.enhancement_layers.add_module("H{}".format(i + 1), el)
        self.ol = Output(
            n1_feature_node_dim * n2_feature_mapping_group
            + n3_enhancement_dim * n4_enhancement_mapping_group,
            out_features,
            c=out_layer_reg,
        )

    def forward(self, *args):
        x, y = None, None
        if self.training:
            x, y = args
        else:
            x = args[0]
        feature_out = None
        for idx, m in enumerate(self.feature_layers):
            if idx == 0:
                feature_out = m(x, y)
            else:
                feature_out = torch.hstack([feature_out, m(x, y)])
        feature_out = self.feature_activation(feature_out)
        enhancement_out = None
        for idx, enhancement in enumerate(self.enhancement_layers):
            if idx == 0:
                enhancement_out = enhancement(feature_out, None)
            else:
                enhancement_out = torch.hstack(
                    [enhancement_out, enhancement(feature_out, None)]
                )
        enhancement_out = self.enhancement_activation(enhancement_out)
        input_of_output_layer = torch.hstack([feature_out, enhancement_out])
        if self.training:
            output = self.ol(input_of_output_layer, y)
        else:
            output = self.ol(input_of_output_layer)
        return output

    def transform(self, x):
        super().transform(x)
        weight = self.ol.weight.data.detach()
        return x @ weight.T


class BasicKernelBLS(nn.Module, AutoEncoder, LatentOutDim):
    def __init__(
            self,
            in_features,
            out_features,
            kernel_dim,
            sigma,
            n_feature_dim,
            n_feature_group,
            n_enhancement_dim,
            n_enhancement_group,
            n_feature_cascade_layer,
            n_enhancement_cascade_layer,
            enhancement_s=0.8,
            output_reg=5 ** -1,
            feature_activation=nn.Sigmoid(),
            enhancement_activation=nn.Sigmoid(),
            feature_cascade_activation=ACTIVATION.none,
            enhancement_cascade_activation=ACTIVATION.none,
            use_positional=False,
            max_len=0,
            auto_encoder=True,
            conv_feature_builder=False,
            device=None,
    ):
        super(BasicKernelBLS, self).__init__()
        if device is None:
            import warnings
            device = torch.device("cpu")
            warnings.warn("device is none", UserWarning)
        if use_positional:
            if in_features % 2 != 0:
                raise ValueError("use positional, in feature should be %2 == 0")
            if max_len == 0:
                raise ValueError("max_len == 0 , ERROR")
            self.pe = PositionalEmbedding(in_features, max_len)
        else:
            self.pe = None
        self.feature_activation = feature_activation
        self.enhancement_activation = enhancement_activation
        self.auto_encoder = auto_encoder
        self.n_feature_group = n_feature_group
        self.n_feature_dim = n_feature_dim
        self.in_features = in_features
        self.n_enhancement_group = n_enhancement_group
        self.n_enhancement_dim = n_enhancement_dim
        self.enhancement_s = enhancement_s
        self.device = device
        self.out_features = out_features
        self.conv_feature_builder = conv_feature_builder
        self.output_reg = output_reg
        if n_feature_cascade_layer > 1:
            self.feature_layer = CascadeKernelFeatureGroup(
                in_features,
                n_feature_dim,
                kernel_dim, sigma,
                n_feature_cascade_layer,
                n_feature_group,
                cascade_activation=feature_cascade_activation,
                device=device,
            )
        else:
            self.feature_layer = KernelFeatureGroup(
                in_features,
                n_feature_dim,
                kernel_dim, sigma,
                n_feature_group,
                bias=True,
                device=device,
            )
        if n_enhancement_cascade_layer > 1:
            self.enhancement_layer = CascadeEnhancementGroup(
                self.feature_layer.get_out_dim(),
                n_enhancement_dim,
                n_enhancement_cascade_layer,
                n_enhancement_group,
                enhancement_s=enhancement_s,
                cascade_activation=enhancement_cascade_activation,
                device=device,
            )
        else:
            self.enhancement_layer = EnhancementGroup(
                self.feature_layer.get_out_dim(),
                n_enhancement_dim,
                n_enhancement_group,
                s=enhancement_s,
                device=device,
            )
        out_in_dim = (
                self.feature_layer.get_out_dim() + self.enhancement_layer.get_out_dim()
        )
        self.output_layer = Output(out_in_dim, out_features, output_reg)
        self.latent_out_dim = self.output_layer.in_features
        self.A = None
        self.A_I = None

    def forward(self, *args):
        x, y = None, None
        if self.training:
            if len(args) > 1:
                x, y = args
            else:
                x = y = args[0]
                
        else:
            x = y = args[0]
        if self.pe:
            x = self.pe(x)
            if self.auto_encoder:
                y = self.pe(y)
                pass
        feature_out = self.feature_layer(x, y)
        feature_out = self.feature_activation(feature_out)
        enhancement_out = self.enhancement_layer(feature_out, y)
        enhancement_out = self.enhancement_activation(enhancement_out)
        self.latent_out_dim = self.enhancement_layer.get_out_dim()
        input_of_output = torch.hstack([feature_out, enhancement_out])
        if self.A is None:
            self.A = input_of_output
            self.A_I = self.output_layer.pinv(input_of_output, self.output_layer.c)
        z = input_of_output
        res = self.output_layer(z, y)
        return res


class BasicBLS(nn.Module, AutoEncoder, LatentOutDim):
    def __init__(
            self,
            in_features,
            out_features,
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
            device=None,
    ):
        super(BasicBLS, self).__init__()
        if device is None:
            import warnings
            device = torch.device("cpu")
            warnings.warn("device is none", UserWarning)
        if use_positional:
            if in_features % 2 != 0:
                raise ValueError("use positional, in feature should be %2 == 0")
            if max_len == 0:
                raise ValueError("max_len == 0 , ERROR")
            self.pe = PositionalEmbedding(in_features, max_len)
        else:
            self.pe = None
        self.feature_activation = feature_activation
        self.enhancement_activation = enhancement_activation
        self.auto_encoder = auto_encoder
        self.n_feature_group = n_feature_group
        self.n_feature_dim = n_feature_dim
        self.in_features = in_features
        self.n_enhancement_group = n_enhancement_group
        self.n_enhancement_dim = n_enhancement_dim
        self.enhancement_s = enhancement_s
        self.device = device
        self.out_features = out_features
        self.conv_feature_builder = conv_feature_builder
        self.output_reg = output_reg
        if n_feature_cascade_layer > 1:
            self.feature_layer = CascadeFeatureGroup(
                in_features,
                n_feature_dim,
                n_feature_cascade_layer,
                n_feature_group,
                cascade_activation=feature_cascade_activation,
                device=device,
            )
        else:
            self.feature_layer = FeatureGroup(
                in_features,
                n_feature_dim,
                n_feature_group,
                bias=True,
                device=device,
            )
        if n_enhancement_cascade_layer > 1:
            self.enhancement_layer = CascadeEnhancementGroup(
                self.feature_layer.get_out_dim(),
                n_enhancement_dim,
                n_enhancement_cascade_layer,
                n_enhancement_group,
                enhancement_s=enhancement_s,
                cascade_activation=enhancement_cascade_activation,
                device=device,
            )
        else:
            self.enhancement_layer = EnhancementGroup(
                self.feature_layer.get_out_dim(),
                n_enhancement_dim,
                n_enhancement_group,
                s=enhancement_s,
                device=device,
            )
        out_in_dim = (
                self.feature_layer.get_out_dim() + self.enhancement_layer.get_out_dim()
        )
        self.output_layer = Output(out_in_dim, out_features, output_reg)
        self.latent_out_dim = self.output_layer.in_features
        self.A = None
        self.A_I = None

    def forward(self, *args):
        x, y = None, None
        if self.training:
            if len(args) > 1:
                x, y = args
            else:
                x = y = args[0]
        else:
            x = y = args[0]
        if self.pe:
            x = self.pe(x)
            if self.auto_encoder:
                y = self.pe(y)
        feature_out = self.feature_layer(x, y)
        feature_out = self.feature_activation(feature_out)
        enhancement_out = self.enhancement_layer(feature_out, y)
        enhancement_out = self.enhancement_activation(enhancement_out)
        self.latent_out_dim = self.enhancement_layer.get_out_dim()
        input_of_output = torch.hstack([feature_out, enhancement_out])
        if self.A is None:
            self.A = input_of_output
            self.A_I = self.output_layer.pinv(input_of_output, self.output_layer.c)
        z = input_of_output
        res = self.output_layer(z, y)
        return res


class KernelBLSBlock(nn.Module):
    def __init__(
            self,
            patch_num,
            patch_size,
            kernel_dim,
            sigma,
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
            device=None,
    ) -> None:
        super().__init__()
        self.patch_num = patch_num
        self.patch_size = patch_size
        self.net = BasicKernelBLS(
            patch_size,
            patch_size,
            kernel_dim,
            sigma,
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
            device,
        )

    def forward(self, x, y=None):
        x1 = patch_fix(x, self.patch_size)
        res = self.net(x1, x1)
        res = patch2series(res, len(x))
        return res

    def decision_function(self, x):
        assert self.training is False, "model is not in eval state"
        assert self.net.training is False, "net model is not in eval state"
        x1 = patch_fix(x, self.patch_size)
        x2 = self.net(x1)
        s = (x2 - x1) ** 2
        s = torch.sqrt(s)
        s = patch2series(s, len(x))
        return s


class BasicBLSBlock(nn.Module):
    def __init__(
            self,
            patch_num,
            patch_size,
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
            device=None,
    ) -> None:
        super().__init__()
        self.patch_num = patch_num
        self.patch_size = patch_size
        self.net = BasicBLS(
            patch_size,
            patch_size,
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
            device,
        )

    def forward(self, x, y=None):
        x1 = patch_fix(x, self.patch_size)
        res = self.net(x1, x1)
        res = patch2series(res, len(x))
        return res

    def decision_function(self, x):
        assert self.training is False, "model is not in eval state"
        assert self.net.training is False, "net model is not in eval state"
        x1 = patch_fix(x, self.patch_size)
        x2 = self.net(x1)
        s = (x2 - x1) ** 2
        s = torch.sqrt(s)
        s = patch2series(s, len(x))
        return s
