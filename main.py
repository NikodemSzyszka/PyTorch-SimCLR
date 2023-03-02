import argparse

parser = argparse.ArgumentParser(description='Growing Tree Hierarchy')
parser.add_argument('--dir', default='.', help='path to dir')
parser.add_argument('--dataset_name', default='mnist', help='dataset name', choices=['mnist', 'fmnist'])
parser.add_argument('--batch_size', default=256, type=int, help='Mini-batch size (default: 256)')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs (default: 100)')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate (default: 0.01)')


def main():
    args = parser.parse_args()
    print(args)

if __name__ == "__main__":
    main()
    