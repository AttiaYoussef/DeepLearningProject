import torch
torch.set_grad_enabled(False)


class SGD(object):
    def __init__(self, model_params, lr, momentum = 0):
        self.model_params = model_params
        self.lr = lr
        self.momentum = momentum