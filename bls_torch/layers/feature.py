from torch import Tensor
from torch.nn import Module, ModuleList
from torch.nn import init
from torch.nn.parameter import Parameter
from .outdim import OutDim
from ..utils.activation import *
import torch
import torch.nn as nn
import numpy as np


class Feature(Module, OutDim):
    
    """
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``
    """
    
    def get_out_dim(self):
        return self.out_features

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None) -> None:
        super(Feature, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.device = device
        if bias:
            self.weight = Parameter(torch.empty(in_features + 1, out_features, device=device),
                                    requires_grad=False)
        else:
            raise Exception('to be implemented')
        # The weight matrix and the bias matrix are randomly generated.
        self.reset_parameters()
        self.max_, self.min_ = None, None   # normalization parameters

    def reset_parameters(self) -> None:
        init.normal_(self.weight)

    def forward(self, *input: Tensor) -> Tensor:
        if self.training:
            input, _ = input
        else:
            input = input[0]
        input_with_bias = torch.hstack(
            [input, 0.1 * torch.ones(input.shape[0], 1, device=input.device, dtype=input.dtype)])
        # During training, optimize weights with SAE; during testing, forward-calculate the feature layer output.
        if self.training:
            if self.bias:
                z = input_with_bias @ self.weight
            else:
                raise Exception('error: undefined method')
            z_min, z_max = torch.min(z, dim=0, keepdim=True).values, torch.max(z, dim=0, keepdim=True).values
            z_scaled = (z - z_min) / (z_max - z_min + 1e-7)
            weight = self.sparse_bls(z_scaled, input_with_bias).T
            self.weight.data = weight   # Update feature layer weight
        # Compute sparse feature
        if self.bias:
            out = input_with_bias @ self.weight
        else:
            raise Exception('error: undefined method')
        # Normalize during training; during testing, use the training-derived maximum and minimum values for normalization.
        if self.training:
            self.max_, self.min_ = torch.max(out, dim=0, keepdim=True).values, torch.min(out, dim=0,
                                                                                         keepdim=True).values
        out = (out - self.min_) / (self.max_ - self.min_ + 1e-7)
        # output tensor
        return out

    @staticmethod
    def sparse_bls(a, b):
        lam = 0.001
        iterations = 50
        aa = a.T @ a
        m = a.shape[1]
        n = b.shape[1]
        x1 = torch.zeros([m, n], device=a.device, dtype=a.dtype)
        wk = x1
        ok = x1
        uk = x1
        l1 = (aa + torch.eye(m, device=a.device)).inverse()
        l2 = (l1 @ a.T) @ b
        for i in range(iterations):
            ck = l2 + (l1 @ (ok - uk))
            ok = torch.maximum(ck + uk - lam, torch.zeros_like(ck + uk - lam, device=a.device)) - torch.maximum(
                -ck - uk - lam, torch.zeros_like(-ck - uk - lam, device=a.device))
            uk = uk + ck - ok
            wk = ok
        return wk

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class KernelProcessing(nn.Module):

    def __init__(self, dimensions=3, sigma=1.0):
        super(KernelProcessing, self).__init__()
        self.dimensions = dimensions
        self.sigma = sigma

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        X = x1
        n_samples, n_features = X.shape
        # Sampling the weight w from the normal distribution.
        w = torch.normal(mean=0.0, std=float(torch.sqrt(torch.tensor(self.sigma))),
                         size=(n_features, self.dimensions)).to(X.device)
        # Sampling the offset b from the uniform distribution.
        b = 2 * np.pi * torch.rand(size=(n_features, self.dimensions)).to(X.device)
        X_expanded = X.unsqueeze(-1).expand(-1, -1, self.dimensions)
        projection = X_expanded * w.unsqueeze(0) + b.unsqueeze(0)
        transformed_features = torch.sqrt(torch.tensor(2.0 / self.dimensions)) * torch.cos(projection)
        transformed_features = transformed_features.view(n_samples, -1)  # Reshape to (n_samples, n_features * dimensions)
        return transformed_features, w, b


class KernelFeature(Module, OutDim):

    """
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``
    """

    def get_out_dim(self):
        return self.out_features * self.dimensions

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, kernel_dim: int, sigma: int, bias: bool = True, device=None) -> None:
        super(KernelFeature, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.device = device
        if bias:
            self.weight = Parameter(torch.empty(in_features + 1, out_features, device=device), requires_grad=False)
        else:
            raise Exception('to be implemented')
        # The weight matrix and the bias matrix are randomly generated.
        self.reset_parameters()
        self.max_, self.min_ = None, None   # normalization parameters
        self.dimensions = kernel_dim
        self.sigma = sigma
        self.kernel_after = KernelProcessing(self.dimensions, self.sigma)
        self.w = None
        self.b = None

    def reset_parameters(self) -> None:
        init.normal_(self.weight)

    def forward(self, *input: Tensor) -> Tensor:
        """
        Args:
        *input: input tensor
        """
        if self.training:
            input, _ = input
        else:
            input = input[0]
        input_with_bias = torch.hstack(
            [input, 0.1 * torch.ones(input.shape[0], 1, device=input.device, dtype=input.dtype)])
        # During training, optimize weights with SAE; during testing, forward-calculate the feature layer output.
        if self.training:
            if self.bias:
                z = input_with_bias @ self.weight
            else:
                raise Exception('error: undefined method')
            z_min, z_max = torch.min(z, dim=0, keepdim=True).values, torch.max(z, dim=0, keepdim=True).values
            z_scaled = (z - z_min) / (z_max - z_min + 1e-7)
            weight = self.sparse_bls(z_scaled, input_with_bias).T
            self.weight.data = weight   # Update feature layer weight
        # Compute sparse feature
        if self.bias:
            out = input_with_bias @ self.weight
        else:
            raise Exception('error: undefined method')
        # Normalize during training; during testing, use the training-derived maximum and minimum values for normalization.
        if self.training:
            self.max_, self.min_ = torch.max(out, dim=0, keepdim=True).values, torch.min(out, dim=0,
                                                                                         keepdim=True).values
        out = (out - self.min_) / (self.max_ - self.min_ + 1e-7)
        if self.training:
            out, w, b = self.kernel_after(out, out)
            self.w = w
            self.b = b
        else:
            n_samples, n_features = out.shape
            w, b = self.w, self.b
            X_expanded = out.unsqueeze(-1).expand(-1, -1, self.dimensions)
            projection = X_expanded * w.unsqueeze(0) + b.unsqueeze(0)
            transformed_features = torch.sqrt(torch.tensor(2.0 / self.dimensions)) * torch.cos(projection)
            out = transformed_features.view(n_samples, -1)
        self.out_features = out.size()[1]
        # output tensor
        return out

    @staticmethod
    def sparse_bls(a, b):
        lam = 0.001
        iterations = 50
        aa = a.T @ a
        m = a.shape[1]
        n = b.shape[1]
        x1 = torch.zeros([m, n], device=a.device, dtype=a.dtype)
        wk = x1
        ok = x1
        uk = x1
        eps = 1e-7
        try:
            l1 = (aa + torch.eye(m, device=a.device) + eps).inverse()
        except:
            l1 = torch.pinverse(aa + torch.eye(m, device=a.device) + eps)
        l2 = (l1 @ a.T) @ b
        for i in range(iterations):
            ck = l2 + (l1 @ (ok - uk))
            ok = torch.maximum(ck + uk - lam, torch.zeros_like(ck + uk - lam, device=a.device)) - torch.maximum(
                -ck - uk - lam, torch.zeros_like(-ck - uk - lam, device=a.device))
            uk = uk + ck - ok
            wk = ok
        return wk

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features * self.dimensions, self.bias is not None
        )


class FeatureGroup(Module, OutDim):

    def get_out_dim(self):
        return self.out_dim

    def __init__(self, in_features: int, out_features: int, n_group=10,
                 bias: bool = True, device=None):
        super(FeatureGroup, self).__init__()
        self.feature_layers = ModuleList()
        self.bias = bias
        self.n_group = n_group
        self.device = device
        for i in range(n_group):
            fl = Feature(in_features, out_features, bias, device)
            self.feature_layers.add_module('FeatureGroup{}'.format(i + 1), fl)
        self.out_dim = out_features * n_group

    def forward(self, *args):
        x, y = None, None
        if self.training:
            x, y = args
        else:
            x = args[0]
        feature_out = None
        for idx, module in enumerate(self.feature_layers):
            res = module(x, y)
            if idx == 0:
                feature_out = res
            else:
                feature_out = torch.hstack([feature_out, res])
        return feature_out


class KernelFeatureGroup(Module, OutDim):

    def get_out_dim(self):
        return self.out_dim

    def __init__(self, in_features: int, out_features: int, kernel_dim: int, sigma: int, n_group=10,
                 bias: bool = True, device=None):
        super(KernelFeatureGroup, self).__init__()
        self.feature_layers = ModuleList()
        self.n_group = n_group
        self.bias = bias
        self.device = device
        self.out_dim = 0
        for i in range(n_group):
            fl = KernelFeature(in_features, out_features, kernel_dim, sigma, bias, device)
            tmp = fl.get_out_dim()
            self.out_dim += tmp
            self.feature_layers.add_module('KernelFeatureGroup{}'.format(i + 1), fl)

    def forward(self, *args):
        x, y = None, None
        if self.training:
            x, y = args
        else:
            x = args[0]
        feature_out = None
        for idx, module in enumerate(self.feature_layers):
            res = module(x, y)
            if idx == 0:
                feature_out = res
            else:
                feature_out = torch.hstack([feature_out, res])
        return feature_out


class CascadeFeatureLayer(Module, OutDim):

    def get_out_dim(self):
        return self.out_dim

    def __init__(self, in_features, hidden_size, num_layer, cascade_activation=ACTIVATION.none, device=None):
        super(CascadeFeatureLayer, self).__init__()
        self.modulelist = ModuleList()
        self.out_dim = hidden_size * num_layer
        self.cascade_activation = cascade_activation
        self.device = device
        for i in range(num_layer):
            self.modulelist.add_module('CFeature{}'.format(i + 1), Feature(
                in_features=in_features if i == 0 else hidden_size,
                out_features=hidden_size,
                device=self.device
            ))

    def forward(self, *args):
        x, y = None, None
        if self.training:
            x, y = args
        else:
            x = args[0]
        ret = None
        for idx, m in enumerate(self.modulelist):
            x = m(x, y)
            x = self.cascade_activation(x)
            if idx == 0:
                ret = x
            else:
                ret = torch.hstack([ret, x])
        return ret


class CascadeFeatureGroup(Module, OutDim):
    
    def get_out_dim(self):
        return self.out_dim

    def __init__(self, in_features, hidden_size, num_layer, num_group, cascade_activation=ACTIVATION.none,
                 device=None):
        super(CascadeFeatureGroup, self).__init__()
        self.modulelist = ModuleList()
        self.out_dim = hidden_size * num_layer * num_group
        self.cascade_activation = cascade_activation
        self.device = device
        for i in range(num_group):
            self.modulelist.add_module('CFeatureGroup{}'.format(i + 1), CascadeFeatureLayer(
                in_features=in_features,
                hidden_size=hidden_size,
                num_layer=num_layer,
                cascade_activation=self.cascade_activation,
                device=device
            ))

    def forward(self, *args):
        x, y = None, None
        if self.training:
            x, y = args
        else:
            x = args[0]
        ret = None
        for idx, m in enumerate(self.modulelist):
            res = m(x, y)
            if idx == 0:
                ret = res
            else:
                ret = torch.hstack([ret, res])
        return ret


class CascadeKernelFeatureLayer(Module, OutDim):

    def get_out_dim(self):
        return self.out_dim

    def __init__(self, in_features, hidden_size, kernel_dim, sigma, num_layer, cascade_activation=ACTIVATION.none,
                 device=None):
        super(CascadeKernelFeatureLayer, self).__init__()
        self.modulelist = ModuleList()
        self.out_dim = 0
        self.cascade_activation = cascade_activation
        self.device = device
        for i in range(num_layer):
            f = KernelFeature(
                in_features=in_features if i == 0 else hidden_size * kernel_dim,
                out_features=hidden_size,
                kernel_dim=kernel_dim, sigma=sigma,
                device=self.device
            )
            self.modulelist.add_module('CKerFeature{}'.format(i + 1), f)
            self.out_dim += f.get_out_dim()

    def forward(self, *args):
        x, y = None, None
        if self.training:
            x, y = args
        else:
            x = args[0]
        ret = None
        for idx, m in enumerate(self.modulelist):
            x = m(x, y)
            x = self.cascade_activation(x)
            if idx == 0:
                ret = x
            else:
                ret = torch.hstack([ret, x])
        return ret


class CascadeKernelFeatureGroup(Module, OutDim):

    def get_out_dim(self):
        return self.out_dim

    def __init__(self, in_features, hidden_size, kernel_dim, sigma, num_layer, num_group,
                 cascade_activation=ACTIVATION.none,
                 device=None):
        super(CascadeKernelFeatureGroup, self).__init__()
        self.modulelist = ModuleList()
        self.out_dim = 0  # hidden_size * num_layer * num_group
        self.cascade_activation = cascade_activation
        self.device = device
        for i in range(num_group):
            f = CascadeKernelFeatureLayer(
                in_features=in_features,
                hidden_size=hidden_size,
                kernel_dim=kernel_dim, sigma=sigma,
                num_layer=num_layer,
                cascade_activation=self.cascade_activation,
                device=device
            )
            self.out_dim += f.get_out_dim()
            self.modulelist.add_module('CKernelFeatureGroup{}'.format(i + 1), f)

    def forward(self, *args):
        x, y = None, None
        if self.training:
            x, y = args
        else:
            x = args[0]
        ret = None
        for idx, m in enumerate(self.modulelist):
            res = m(x, y)
            if idx == 0:
                ret = res
            else:
                ret = torch.hstack([ret, res])
        return ret
