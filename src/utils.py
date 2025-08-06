import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset, Subset
import numpy as np

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def load_datasets(num_clients, reference_size):
    """
    Loads CIFAR-10 and partitions it for clients and a server reference set.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Split training data for clients (IID for now)
    num_items = int(len(trainset) / num_clients)
    client_datasets, all_idxs = {}, [i for i in range(len(trainset))]
    for i in range(num_clients):
        client_idxs = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - client_idxs)
        client_datasets[i] = DatasetSplit(trainset, client_idxs)
    
    # Create a small, clean reference dataset for the server from the test set
    all_test_idxs = list(range(len(testset)))
    reference_idxs = np.random.choice(all_test_idxs, reference_size, replace=False)
    reference_dataset = Subset(testset, reference_idxs)
    
    # Create data loaders
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)
    reference_loader = DataLoader(reference_dataset, batch_size=32, shuffle=False)
    
    return client_datasets, test_loader, reference_loader