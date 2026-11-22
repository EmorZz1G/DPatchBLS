import torch
from torch import Tensor
from torch.nn import init
from torch.nn import Module, ModuleList
from torch.nn.parameter import Parameter
import torch.nn as nn
from .outdim import OutDim, InputToDeviceHookMixin, Input2DoubleTensorHookMixin
from ..utils.activation import ACTIVATION


class Enhancement(Module, OutDim):
    
    """
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: shrink coefficient
        bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``.
    """
    
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def get_out_dim(self):
        return self.out_features

    def __init__(self, in_features: int, out_features: int, s: float = 0.8, bias: bool = True,
                 device=None) -> None:
        super(Enhancement, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.bias = bias
        self.device = device
        if self.bias:
            self.weight = Parameter(torch.empty(self.in_features+1, self.out_features, device=device), requires_grad=False)
        else:
            raise Exception('to be implemented')
        self.reset_parameters()
        self.parameterOfShrink = None

    @staticmethod
    def orth(weight: Tensor) -> Tensor:
        r = torch.linalg.matrix_rank(weight)
        u, s, v = torch.svd(weight)
        return u[:, :r]

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        in_features, out_features = self.weight.shape[0] - 1, self.weight.shape[1]
        if in_features >= out_features:
            self.weight.data = self.orth(self.weight)
        else:
            self.weight.data = self.orth(self.weight.T).T

    def forward(self, *input: Tensor) -> Tensor:
        """
        Computes the output of the feedforward enhancement layer.
        :param input: Input tensor
        :return: Output tensor
        """
        if self.training:
            input, _ = input
        else:
            input = input[0]
        input_with_bias = torch.hstack([input, 0.1 * torch.ones(input.shape[0], 1, device=input.device,
                                                                dtype=input.dtype)])
        if self.bias:
            out = input_with_bias @ self.weight
        else:
            raise Exception('to be implemented')
        # The training stage calculates scaling coefficients, which are then used in the testing stage.
        if self.training:
            self.parameterOfShrink = self.s / torch.max(out)
        else:
            pass
        out = out * self.parameterOfShrink
        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, s={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.s
        )


class EnhancementGroup(Module, OutDim, InputToDeviceHookMixin, Input2DoubleTensorHookMixin):

    def get_out_dim(self):
        return self.out_dim

    def __init__(self, in_features: int, out_features: int, n_group = 10,
                 s: float = 0.8, bias: bool = True, device = None):
        super(EnhancementGroup, self).__init__()
        self.enhancement_layers = nn.ModuleList()
        self.in_features = in_features
        self.device = device
        self.out_features = out_features
        self.n_group = n_group
        self.s = s
        self.bias = bias
        self.out_dim = self.n_group * self.out_features
        for i in range(n_group):
            el = Enhancement(in_features, out_features, s, bias, device)
            self.enhancement_layers.add_module('EnhancementGroup{}'.format(i + 1), el)
        InputToDeviceHookMixin.hook(self)
        Input2DoubleTensorHookMixin.hook(self)

    def forward(self, *args):
        x, y = None, None
        if self.training:
            x, y = args
        else:
            x = args[0]
        enhancement_out = None
        for idx, module in enumerate(self.enhancement_layers):
            res = module(x, y)
            if idx == 0:
                enhancement_out = res
            else:
                enhancement_out = torch.hstack([enhancement_out, res])
        return enhancement_out


class CascadeEnhancementLayer(Module, OutDim):

    def get_out_dim(self):
        return self.hidden_size * self.num_layer

    def __init__(self, in_features, hidden_size, num_layer,
                 enhancement_s = 0.8,
                 cascade_activation = ACTIVATION.none,
                 device = None):
        super(CascadeEnhancementLayer, self).__init__()
        self.modulelist = ModuleList()
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.cascade_activation = cascade_activation
        self.device = device
        for i in range(num_layer):
            self.modulelist.add_module('CEnhancement{}'.format(i + 1), Enhancement(
                in_features=in_features if i == 0 else hidden_size,
                out_features=hidden_size,
                s=enhancement_s,
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


class CascadeEnhancementGroup(Module, OutDim, InputToDeviceHookMixin, Input2DoubleTensorHookMixin):

    def get_out_dim(self):
        return self.hidden_size * self.num_layer * self.num_group

    def __init__(self, in_features, hidden_size, num_layer, num_group,
                 enhancement_s = 0.8,
                 cascade_activation = ACTIVATION.none,
                 device = None):
        super(CascadeEnhancementGroup, self).__init__()
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.num_group = num_group
        self.cascade_activation = cascade_activation
        self.device = device
        self.modulelist = ModuleList()
        for i in range(num_group):
            self.modulelist.add_module('CEnhancementGroup{}'.format(i + 1), CascadeEnhancementLayer(
                in_features=in_features,
                hidden_size=hidden_size,
                num_layer=num_layer,
                enhancement_s=enhancement_s,
                cascade_activation=self.cascade_activation,
                device=device
            ))
        InputToDeviceHookMixin.hook(self)
        Input2DoubleTensorHookMixin.hook(self)

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
