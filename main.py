<<<<<<< HEAD
import torch
from src.utils import load_datasets
from src.client import Client
from src.server import Server
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# --- Simulation Parameters ---
NUM_CLIENTS = 10
NUM_ROUNDS = 20
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.01
MALICIOUS_FRACTION = 0.3
SPARSITY = 0.05
REFERENCE_DATA_SIZE = 200

def plot_reputation_history(reputation_history, malicious_ids):
    # (This function remains unchanged)
    reputation_history = np.array(reputation_history)
    plt.figure(figsize=(12, 8))
    for i in range(reputation_history.shape[1]):
        if i in malicious_ids:
            plt.plot(reputation_history[:, i], label=f'Malicious Client {i}', linestyle='--', marker='x')
        else:
            plt.plot(reputation_history[:, i], label=f'Benign Client {i}', marker='o', alpha=0.7)
    plt.title('Client Reputation Over Time')
    plt.xlabel('Training Round')
    plt.ylabel('Reputation Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reputation_plot.png')
    plt.close()

def plot_tsne(client_updates_2d, malicious_ids, round_num, x_lims, y_lims):
    """
    Creates a 2D t-SNE plot with fixed axis limits.
    """
    plt.figure(figsize=(10, 8))
    
    benign_indices = [i for i in range(len(client_updates_2d)) if i not in malicious_ids]
    if benign_indices:
        plt.scatter(client_updates_2d[benign_indices, 0], client_updates_2d[benign_indices, 1], 
                    marker='o', s=100, c='blue', alpha=0.7, label='Benign Clients')

    malicious_plot_indices = [i for i in range(len(client_updates_2d)) if i in malicious_ids]
    if malicious_plot_indices:
        plt.scatter(client_updates_2d[malicious_plot_indices, 0], client_updates_2d[malicious_plot_indices, 1], 
                    marker='x', s=100, c='red', label='Malicious Clients')
            
    plt.legend()
    plt.title(f't-SNE Visualization of Client Updates (Round {round_num})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    # Apply the global axis limits
    plt.xlim(x_lims)
    plt.ylim(y_lims)
    plt.grid(True)
    plt.tight_layout()
    # Save in a dedicated folder
    if not os.path.exists('tsne_plots'):
        os.makedirs('tsne_plots')
    plt.savefig(f'tsne_plots/tsne_round_{round_num}.png')
    plt.close()

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    client_datasets, test_loader, reference_loader = load_datasets(NUM_CLIENTS, REFERENCE_DATA_SIZE)

    num_malicious = int(NUM_CLIENTS * MALICIOUS_FRACTION)
    malicious_ids = np.random.choice(np.arange(NUM_CLIENTS), num_malicious, replace=False)
    print(f"Designating clients {malicious_ids} as malicious.")
    
    clients = [Client(i, client_datasets[i], True if i in malicious_ids else False) for i in range(NUM_CLIENTS)]
    server = Server(num_clients=NUM_CLIENTS, test_loader=test_loader, reference_loader=reference_loader)

    # --- Data Storage ---
    reputation_history = []
    update_history = []

    print("\nStarting FL Simulation...")
    for round_num in range(NUM_ROUNDS):
        print(f"\n--- Round {round_num + 1}/{NUM_ROUNDS} ---")
        global_model = server.global_model
        client_updates = []
        for client in clients:
            indices, values = client.train(global_model, LOCAL_EPOCHS, LEARNING_RATE, SPARSITY)
            client_updates.append((indices, values))
        
        reconstructed_deltas = server.aggregate_updates(client_updates)
        accuracy = server.evaluate()
        
        reputation_history.append(server.client_reputations.cpu().numpy().copy())
        update_history.append(torch.stack(reconstructed_deltas).cpu().numpy())

        print(f"Global Model Accuracy: {accuracy:.2f}%")

    print("\nFederated Learning Simulation Finished!")
    
    # --- Generate Visualizations ---
    print("Generating visualizations...")
    
    # 1. Generate final reputation plot
    plot_reputation_history(reputation_history, malicious_ids)

    # 2. Process all updates to get global axis limits for t-SNE
    print("Processing t-SNE plots with fixed axes...")
    all_updates_flat = np.concatenate(update_history)
    tsne = TSNE(n_components=2, perplexity=min(30, len(all_updates_flat)-1), random_state=42, n_iter=1000)
    all_updates_2d = tsne.fit_transform(all_updates_flat)
    
    # Find the global min/max for x and y axes
    x_lims = (all_updates_2d[:, 0].min() - 5, all_updates_2d[:, 0].max() + 5)
    y_lims = (all_updates_2d[:, 1].min() - 5, all_updates_2d[:, 1].max() + 5)

    # 3. Generate a t-SNE plot for each round using the fixed limits
    start_index = 0
    for i, round_updates in enumerate(update_history):
        num_clients_this_round = len(round_updates)
        round_updates_2d = all_updates_2d[start_index : start_index + num_clients_this_round]
        plot_tsne(round_updates_2d, malicious_ids, i + 1, x_lims, y_lims)
        start_index += num_clients_this_round

    print("Visualizations saved. Check the 'tsne_plots' folder.")

if __name__ == "__main__":
=======
import torch
from src.utils import load_datasets
from src.client import Client
from src.server import Server
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# --- Simulation Parameters ---
NUM_CLIENTS = 10
NUM_ROUNDS = 20
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.01
MALICIOUS_FRACTION = 0.3
SPARSITY = 0.05
REFERENCE_DATA_SIZE = 200

def plot_reputation_history(reputation_history, malicious_ids):
    # (This function remains unchanged)
    reputation_history = np.array(reputation_history)
    plt.figure(figsize=(12, 8))
    for i in range(reputation_history.shape[1]):
        if i in malicious_ids:
            plt.plot(reputation_history[:, i], label=f'Malicious Client {i}', linestyle='--', marker='x')
        else:
            plt.plot(reputation_history[:, i], label=f'Benign Client {i}', marker='o', alpha=0.7)
    plt.title('Client Reputation Over Time')
    plt.xlabel('Training Round')
    plt.ylabel('Reputation Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reputation_plot.png')
    plt.close()

def plot_tsne(client_updates_2d, malicious_ids, round_num, x_lims, y_lims):
    """
    Creates a 2D t-SNE plot with fixed axis limits.
    """
    plt.figure(figsize=(10, 8))
    
    benign_indices = [i for i in range(len(client_updates_2d)) if i not in malicious_ids]
    if benign_indices:
        plt.scatter(client_updates_2d[benign_indices, 0], client_updates_2d[benign_indices, 1], 
                    marker='o', s=100, c='blue', alpha=0.7, label='Benign Clients')

    malicious_plot_indices = [i for i in range(len(client_updates_2d)) if i in malicious_ids]
    if malicious_plot_indices:
        plt.scatter(client_updates_2d[malicious_plot_indices, 0], client_updates_2d[malicious_plot_indices, 1], 
                    marker='x', s=100, c='red', label='Malicious Clients')
            
    plt.legend()
    plt.title(f't-SNE Visualization of Client Updates (Round {round_num})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    # Apply the global axis limits
    plt.xlim(x_lims)
    plt.ylim(y_lims)
    plt.grid(True)
    plt.tight_layout()
    # Save in a dedicated folder
    if not os.path.exists('tsne_plots'):
        os.makedirs('tsne_plots')
    plt.savefig(f'tsne_plots/tsne_round_{round_num}.png')
    plt.close()

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    client_datasets, test_loader, reference_loader = load_datasets(NUM_CLIENTS, REFERENCE_DATA_SIZE)

    num_malicious = int(NUM_CLIENTS * MALICIOUS_FRACTION)
    malicious_ids = np.random.choice(np.arange(NUM_CLIENTS), num_malicious, replace=False)
    print(f"Designating clients {malicious_ids} as malicious.")
    
    clients = [Client(i, client_datasets[i], True if i in malicious_ids else False) for i in range(NUM_CLIENTS)]
    server = Server(num_clients=NUM_CLIENTS, test_loader=test_loader, reference_loader=reference_loader)

    # --- Data Storage ---
    reputation_history = []
    update_history = []

    print("\nStarting FL Simulation...")
    for round_num in range(NUM_ROUNDS):
        print(f"\n--- Round {round_num + 1}/{NUM_ROUNDS} ---")
        global_model = server.global_model
        client_updates = []
        for client in clients:
            indices, values = client.train(global_model, LOCAL_EPOCHS, LEARNING_RATE, SPARSITY)
            client_updates.append((indices, values))
        
        reconstructed_deltas = server.aggregate_updates(client_updates)
        accuracy = server.evaluate()
        
        reputation_history.append(server.client_reputations.cpu().numpy().copy())
        update_history.append(torch.stack(reconstructed_deltas).cpu().numpy())

        print(f"Global Model Accuracy: {accuracy:.2f}%")

    print("\nFederated Learning Simulation Finished!")
    
    # --- Generate Visualizations ---
    print("Generating visualizations...")
    
    # 1. Generate final reputation plot
    plot_reputation_history(reputation_history, malicious_ids)

    # 2. Process all updates to get global axis limits for t-SNE
    print("Processing t-SNE plots with fixed axes...")
    all_updates_flat = np.concatenate(update_history)
    tsne = TSNE(n_components=2, perplexity=min(30, len(all_updates_flat)-1), random_state=42, n_iter=1000)
    all_updates_2d = tsne.fit_transform(all_updates_flat)
    
    # Find the global min/max for x and y axes
    x_lims = (all_updates_2d[:, 0].min() - 5, all_updates_2d[:, 0].max() + 5)
    y_lims = (all_updates_2d[:, 1].min() - 5, all_updates_2d[:, 1].max() + 5)

    # 3. Generate a t-SNE plot for each round using the fixed limits
    start_index = 0
    for i, round_updates in enumerate(update_history):
        num_clients_this_round = len(round_updates)
        round_updates_2d = all_updates_2d[start_index : start_index + num_clients_this_round]
        plot_tsne(round_updates_2d, malicious_ids, i + 1, x_lims, y_lims)
        start_index += num_clients_this_round

    print("Visualizations saved. Check the 'tsne_plots' folder.")

if __name__ == "__main__":
>>>>>>> 75bc55b7d4e83b4211d53d15e20df6192ca35e65
    main()