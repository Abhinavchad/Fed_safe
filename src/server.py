import torch
from collections import OrderedDict
from .model import SimpleCNN
import copy
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

class Server:
    def __init__(self, num_clients, test_loader, reference_loader):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = SimpleCNN().to(self.device)
        self.num_clients = num_clients
        self.test_loader = test_loader
        self.reference_loader = reference_loader
        self.total_model_params = sum(p.numel() for p in self.global_model.parameters())
        self.client_reputations = torch.ones(num_clients, device=self.device)
        self.reputation_threshold = 0.2 # Lower threshold to only catch persistent attackers

    def _get_loss(self, model):
        model.eval(); criterion = torch.nn.CrossEntropyLoss(); total_loss = 0.0
        with torch.no_grad():
            for images, labels in self.reference_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images); loss = criterion(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(self.reference_loader)

    def aggregate_updates(self, client_updates):
        reconstructed_deltas = []
        for indices, values in client_updates:
            client_delta = torch.zeros(self.total_model_params, device=self.device)
            client_delta[indices] = values.to(self.device)
            reconstructed_deltas.append(client_delta)
        client_scores = []
        for i in range(self.num_clients):
            temp_model = copy.deepcopy(self.global_model)
            with torch.no_grad():
                global_params = torch.cat([p.view(-1) for p in temp_model.parameters()])
                updated_params = global_params + reconstructed_deltas[i]
                offset = 0
                for param in temp_model.parameters(): param.view(-1).copy_(updated_params[offset:offset+param.numel()]); offset += param.numel()
            client_scores.append(self._get_loss(temp_model))
        
        median_loss = np.median(client_scores)
        flagged_clients = {i for i, score in enumerate(client_scores) if score > median_loss * 1.2} # More tolerant loss flagging

        deltas_np = torch.stack(reconstructed_deltas).cpu().numpy()
        if len(deltas_np) > 1:
            distance_matrix = np.maximum(1 - cosine_similarity(deltas_np), 0)
            eps_value = np.median(distance_matrix)
            min_samples = max(2, int(self.num_clients * 0.3)) 
            clustering = DBSCAN(eps=eps_value, min_samples=min_samples, metric="precomputed").fit(distance_matrix)
            outlier_indices = {i for i, label in enumerate(clustering.labels_) if label == -1}
            flagged_clients.update(outlier_indices)
        
        for i in range(self.num_clients):
            if i in flagged_clients:
                self.client_reputations[i] -= 0.3 # Penalize
            else:
                self.client_reputations[i] += 0.1 # Reward honest clients to prevent false positives from killing them
        
        self.client_reputations = torch.clamp(self.client_reputations, 0, 1)
        print(f"Updated reputations: {[f'{r:.2f}' for r in self.client_reputations]}")
        
        # --- Final Aggregation ---
        trusted_client_indices = [i for i, rep in enumerate(self.client_reputations) if rep > self.reputation_threshold]
        print(f"Final trusted aggregation group: {trusted_client_indices}")
        
        if not trusted_client_indices:
            print("No clients trusted. Skipping update.")
            return reconstructed_deltas # Still return deltas even if no update

        # Use standard FedAvg for stability with non-IID data
        trusted_deltas = torch.stack([reconstructed_deltas[i] for i in trusted_client_indices])
        aggregated_delta = torch.mean(trusted_deltas, dim=0)

        # Apply final update
        with torch.no_grad():
            global_params = torch.cat([p.view(-1) for p in self.global_model.parameters()])
            updated_params = global_params + aggregated_delta
            offset = 0
            for param in self.global_model.parameters(): param.view(-1).copy_(updated_params[offset:offset+param.numel()]); offset += param.numel()
        
        # *** The only change is adding this return statement ***
        return reconstructed_deltas

    def evaluate(self):
        # (This method remains unchanged)
        self.global_model.eval(); correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images); _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0); correct += (predicted == labels).sum().item()
        return 100 * correct / total