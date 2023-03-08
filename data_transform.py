from torchvision.transforms import transforms
from torchvision import transforms, datasets

class Datasets:
    def __init__(self, root):
        self.root = root
        self.datasets_train = { 'mnist': lambda: datasets.MNIST(self.root, train=True,
                                                        transform = TwoViews(self.get_data_augmentation()), download=True),
                                'fmnist': lambda: datasets.FashionMNIST(self.root, train=True,
                                                        transform = TwoViews(self.get_data_augmentation()), download=True)                                  
                                }
        self.datasets_valid = { 'mnist': lambda: datasets.MNIST(self.root, train=False,
                                                        transform= transforms.ToTensor(), download=True),
                                'fmnist': lambda: datasets.FashionMNIST(self.root, train=False,
                                                        transform = transforms.ToTensor(), download=True)                                  
                                }
                                
    @staticmethod
    def get_data_augmentation():
        transform_pipeline = transforms.Compose([transforms.RandomResizedCrop(size=32),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                                        transforms.RandomGrayscale(p=0.2),
                                        transforms.GaussianBlur(1, sigma=(0.1, 2.0)),
                                        transforms.ToTensor()])
        return transform_pipeline

    def get_train_dataset(self, dataset_name):
        return self.datasets_train[dataset_name]()
    
    def get_valid_dataset(self, dataset_name):
        return self.datasets_valid[dataset_name]()
    
class TwoViews:
    def __init__(self, transform_pipeline ):
        self.transform_pipeline = transform_pipeline 

    def __call__(self, x):
        return [self.transform_pipeline(x), self.transform_pipeline(x)]
        
                        
    
