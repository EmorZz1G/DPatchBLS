import time
import numpy as np
import os
import torch
import random


def seed_all(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TimeRecorder:

    def __init__(self, device):
        self.device = device
        self.times_tab = {}
        self.last_event = None

    def start(self, event):
        if 'cuda' in self.device:
            torch.cuda.synchronize(self.device)
        self.times_tab[event] = time.perf_counter()
        self.last_event = event

    def stop(self, event=None):
        if event is None:
            event = self.last_event
        if event is None:
            print('no event exists')
            return
        elif event not in self.times_tab:
            print('no event started')
            return
        if 'cuda' in self.device:
            torch.cuda.synchronize(self.device)
        tm = time.perf_counter() - self.times_tab[event]
        return tm
