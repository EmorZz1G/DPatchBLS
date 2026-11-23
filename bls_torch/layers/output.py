import torch
from torch.nn import Module
from torch.nn.parameter import Parameter


class Output(Module):
    
    def __init__(self, in_features: int, out_features: int, c: float) -> None:
        super(Output, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.weight = Parameter(torch.randn([in_features, out_features]))

    def forward(self, *args):
        
        """
        :param args: tensor tuple (x, y) training / x testing
        :return: tensor
        """
        
        if self.training:
            x, y = args
            tmp = self.pinv(x, self.c)
            weight = tmp @ y  # Computation of the Output Layer Weights via Ridge Regression
            self.weight.data = weight
        else:
            x = args[0]
        return x @ self.weight

    def pinv(self, A, reg):
        try:
            ret = (reg * torch.eye(A.shape[1]).to(A.device) + A.T @ A).inverse() @ A.T
        except Exception as e:
            ret = torch.pinverse(reg * torch.eye(A.shape[1], device=A.device, dtype=A.dtype) + A.T @ A) @ A.T
        return ret

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, c={}'.format(self.in_features, self.out_features, self.c)
