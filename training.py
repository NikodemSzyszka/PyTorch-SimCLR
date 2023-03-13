import torch
import torch.nn as nn
from tqdm import tqdm

class TrainingModule():

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        

    def train(self, train_loader):
        for epoch_counter in range(self.args.epochs):
            for i, (images, _) in enumerate(tqdm(train_loader, desc = f"Epoch: {epoch_counter}")):
                pass

