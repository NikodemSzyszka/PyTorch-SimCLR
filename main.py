import argparse
import torch
from data_transform import Datasets
from model import SimCLR
from training import TrainingModule
import torch.backends.cudnn as cudnn
from math import sqrt

parser = argparse.ArgumentParser(description='Growing Tree Hierarchy')
parser.add_argument('--dir', default='.', help='path to dir')
parser.add_argument('--dataset_name', default='mnist', help='dataset name', choices=['mnist', 'fmnist'])
parser.add_argument('--batch_size', default=256, type=int, help='Mini-batch size (default: 256)')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs (default: 100)')
parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate (default: 0.01)')
parser.add_argument('--lr_scaling', default='none', help='dataset name', choices=['none', 'linear', 'sqrt'])
parser.add_argument('--train', default=True, type=bool, help='True - train, False - valid')
parser.add_argument('--temperature', default=0.1, type=float, help='Temperature parameter in loss funcion (default: 0.1)')
parser.add_argument('--weight_decay', default=1e-6, type=float, help='weight decay (default: 1e-6)')
parser.add_argument('--save frequency', default=100, type=int, help='save frequency(default: 100)')

def main():
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    if args.train:
        dataset = Datasets(args.dir)
        train_dataset = dataset.get_train_dataset(args.dataset_name)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                                num_workers=2, pin_memory=True, drop_last=True)
        model = SimCLR()
        model.to(args.device)
        if args.lr_scaling != 'none':
            args.lr = 0.075*sqrt(args.batch_size) if args.lr_scaling == 'sqrt' else 0.3 * args.batch_size/256
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, verbose=True)
        train_module = TrainingModule(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        train_module.train(train_loader)
    else:
        valid_data = dataset.get_valid_dataset(args.dataset_name)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=1, shuffle=True, 
                                                num_workers=2, pin_memory=True, drop_last=True)
    

if __name__ == "__main__":
    main()
    