from torchvision.transforms import transforms
from torchvision import transforms, datasets

class Datasets:
    def __init__(self, root):
        self.root = root
        self.datasets_train = { 'mnist': lambda: datasets.MNIST(self.root, train=True,
                                                        transform= transforms.ToTensor(), download=True),
                                'fmnist': lambda: datasets.FashionMNIST(self.root, train=True,
                                                        transform = transforms.ToTensor(), download=True)                                  
                                }
        self.datasets_valid = { 'mnist': lambda: datasets.MNIST(self.root, train=False,
                                                        transform= transforms.ToTensor(), download=True),
                                'fmnist': lambda: datasets.FashionMNIST(self.root, train=False,
                                                        transform = transforms.ToTensor(), download=True)                                  
                                }

    def get_train_dataset(self, dataset_name):
        return self.datasets_train[dataset_name]()
    
    def get_valid_dataset(self, dataset_name):
        return self.datasets_valid[dataset_name]()

        
                        
    
