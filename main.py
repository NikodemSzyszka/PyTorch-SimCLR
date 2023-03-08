import argparse
import torch
from data_transform import Datasets

parser = argparse.ArgumentParser(description='Growing Tree Hierarchy')
parser.add_argument('--dir', default='.', help='path to dir')
parser.add_argument('--dataset_name', default='mnist', help='dataset name', choices=['mnist', 'fmnist'])
parser.add_argument('--batch_size', default=256, type=int, help='Mini-batch size (default: 256)')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs (default: 100)')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate (default: 0.01)')
parser.add_argument('--train', default=True, type=bool, help='True - train, False - valid')


def main():
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if train:
        dataset = Datasets(args.dir)
        train_data = dataset.get_train_dataset(args.dataset_name)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                                num_workers = 2, pin_memory = True, drop_last=True)
    else:
        valid_data = dataset.get_valid_dataset(args.dataset_name)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=1, shuffle=True, 
                                                num_workers=2, pin_memory=True, drop_last=True)
    

if __name__ == "__main__":
    main()
    