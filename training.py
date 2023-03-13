import torch
import torch.nn as nn
from tqdm import tqdm

class TrainingModule():

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.criterion = nn.CrossEntropyLoss().to(self.args.device)
        
    def loss(self, representations):
        similarity = nn.functional.cosine_similarity(representations.unsqueeze(1), representations, dim = -1)
        return self.criterion(similarity, torch.zeros(similarity.shape[0], dtype = torch.long).to(self.args.device))

    def train(self, train_loader):
        for epoch_counter in range(self.args.epochs):
            for i, (images, _) in enumerate(tqdm(train_loader, desc = f"Epoch: {epoch_counter}")):
                images = torch.cat(images, dim=0).to(self.args.device)
                representations = self.model(images)
                loss = self.loss(representations)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

