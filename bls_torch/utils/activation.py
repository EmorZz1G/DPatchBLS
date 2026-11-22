import torch


class ACTIVATION:
    @staticmethod
    def none(x):
        return x

    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def sigmoid(x):
        return torch.sigmoid(x)

    @staticmethod
    def relu(x):
        return torch.relu(x)

    @staticmethod
    def tanh(x):
        return torch.tanh(x)
