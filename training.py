import torch
import torch.nn as nn
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter

class TrainingModule():

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model']
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = nn.CrossEntropyLoss().to(self.args.device)
        self.masks = self.create_masks(2 * self.args.batch_size)
        self.logs = SummaryWriter(log_dir=f"{self.args.dir}/logs")
    
    def create_masks(self, N):
        idx = torch.cat([torch.arange(N//2, N), torch.arange(N//2)], dim = 0)
        positive_mask = torch.eye(N, dtype = torch.bool)[idx]
        negative_mask = ~torch.eye(N, dtype = torch.bool)[idx]
        negative_mask.fill_diagonal_(False)
        return positive_mask, negative_mask

    def loss(self, representations):
        similarity = nn.functional.cosine_similarity(representations.unsqueeze(1), representations, dim = -1)
        positives = similarity[self.masks[0]].unsqueeze(-1)
        negatives = similarity[self.masks[1]].view(2*self.args.batch_size, -1)
        labels = torch.zeros(2 * self.args.batch_size, dtype = torch.long, device = self.args.device)
        return self.criterion(torch.cat([positives, negatives], dim =1)/self.args.temperature, labels)

    def train(self, train_loader):
        for epoch_counter in range(self.args.epochs):
            for i, (images, _) in enumerate(tqdm(train_loader, desc = f"Epoch: {epoch_counter}")):
                images = torch.cat(images, dim=0).to(self.args.device)
                representations = self.model(images)
                loss = self.loss(representations)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            self.logs.add_scalar('Loss', loss, global_step = epoch_counter)
            self.scheduler.step()
            if epoch_counter % self.args.save_frequency:
                torch.save({'epoch': epoch_counter,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict(),
                            }, f=os.path.join(self.args.dir, f'Simclr_checkpoint_{epoch_counter}.tar'))

        torch.save({'epoch': self.args.epochs,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    }, f=os.path.join(self.args.dir, f'Simclr_checkpoint_FinalModel.tar'))         
