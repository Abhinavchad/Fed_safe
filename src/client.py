import torch
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
from .model import SimpleCNN

class MaliciousLabelDataset(Dataset):
    def __init__(self, original_dataset, source_label, target_label):
        self.original_dataset = original_dataset
        self.source_label = source_label
        self.target_label = target_label

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, item):
        image, label = self.original_dataset[item]
        if label == self.source_label:
            label = self.target_label
        return image, label

class Client:
    def __init__(self, client_id, local_dataset, malicious=False):
        self.client_id = client_id
        self.local_dataset = local_dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.malicious = malicious

    def train(self, global_model, local_epochs, learning_rate, sparsity):
        """
        Trains locally and computes a sparse update.
        """
        # Load the global model's state
        local_model = SimpleCNN().to(self.device)
        local_model.load_state_dict(global_model.state_dict())
        local_model.train()

        # Setup optimizer and loss function
        optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Determine dataset for training
        dataset_to_train = self.local_dataset
        if self.malicious:
            dataset_to_train = MaliciousLabelDataset(self.local_dataset, source_label=3, target_label=5)

        local_dataloader = DataLoader(dataset_to_train, batch_size=32, shuffle=True)

        # Local training loop
        for epoch in range(local_epochs):
            for images, labels in local_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = local_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # --- Create the sparse update ---
        with torch.no_grad():
            # Calculate the update (delta)
            update = OrderedDict()
            for (global_key, global_param), (local_key, local_param) in zip(global_model.state_dict().items(), local_model.state_dict().items()):
                update[global_key] = local_param - global_param

            # Flatten all updates into a single vector
            flat_update = torch.cat([p.view(-1) for p in update.values()])
            
            # Calculate the number of parameters to keep (top-k)
            k = int(len(flat_update) * sparsity)
            
            # Find the indices and values of the top-k largest magnitude changes
            topk_values, topk_indices = torch.topk(torch.abs(flat_update), k)
            
            # We need to send the actual values (with sign), not their absolute values
            sparse_update_values = flat_update[topk_indices]

        return topk_indices, sparse_update_values