import torch
import numpy


class OutDim:
    def get_out_dim(self):
        raise Exception("not implemented out dim")


class LatentOutDim:
    def get_latent_out_dim(self):
        raise Exception("not implemented latent out dim")


def input2double_tensor(mdl, inp):
    inp2 = [None] * len(inp)
    for i in range(len(inp)):
        if isinstance(inp[i], numpy.ndarray):
            inp2[i] = torch.DoubleTensor(inp[i])
        else:
            inp2[i] = inp[i]
    return tuple(inp2)


def input_to_device(mdl, inp):
    inp2 = [None] * len(inp)
    verbose = False
    for i in range(len(inp)):
        if isinstance(inp[i], torch.Tensor):
            if verbose:
                print("input_to_device a", mdl.__class__.__name__, inp[i].device, mdl.device)
            if inp[i].device != mdl.device:
                inp2[i] = inp[i].to(mdl.device)
            else:
                inp2[i] = inp[i]
            if verbose:
                print("input_to_device b", mdl.__class__.__name__, inp2[i].device, mdl.device)
    return tuple(inp2)


class InputToDeviceHookMixin:
    def hook(self, verbose=False):
        if not hasattr(self, "device"):
            raise Exception("module no attr device")
        elif hasattr(self, "register_forward_pre_hook"):
            if verbose:
                print("InputToDeviceHookMixin ok")
            self.register_forward_pre_hook(input_to_device)
        else:
            raise Exception("InputToDeviceHookMixin err")


class Input2DoubleTensorHookMixin:
    def hook(self, verbose=False):
        if hasattr(self, "register_forward_pre_hook"):
            if verbose:
                print("Input2DoubleTensorHookMixin ok")
            self.register_forward_pre_hook(input2double_tensor)
        else:
            raise Exception("Input2DoubleTensorHookMixin err")
