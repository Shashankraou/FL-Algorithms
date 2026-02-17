"""
Decentralized Federated Learning (DFL) - MNIST Benchmark
=========================================================
Implements and compares four decentralized FL algorithms:
  - D-FedAvg  : Baseline gossip-based federated averaging
  - D-FedProx : FedAvg + proximal regularization term to limit client drift
  - D-FedPer  : Personalized FL — shared backbone, personal classification head
  - D-FedMask : Sparse mask-based FL — each client trains a subnetwork

Key idea: There is NO central server. Clients communicate only with their
neighbors according to a topology graph (mesh or ring) via a gossip protocol.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm.auto import tqdm

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# Use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ── Hyperparameters & Configuration ──────────────────────────────────────────
NUM_CLIENTS     = 5       # Number of participating clients (nodes)
BATCH_SIZE      = 128     # Mini-batch size for local SGD
LOCAL_EPOCHS    = 5       # Number of local training epochs per round
GLOBAL_ROUNDS   = 20      # Total number of communication rounds
LR              = 0.01    # SGD learning rate
MOMENTUM        = 0.9     # SGD momentum (used in D-FedAvg)
WEIGHT_DECAY    = 1e-4    # L2 regularization (used in D-FedAvg)
MU              = 0.01    # Proximal term coefficient for D-FedProx
MASK_SPARSITY   = 0.1     # Fraction of weights zeroed out per client in D-FedMask
TOPOLOGY        = "mesh"  # Network topology: "mesh" (all-to-all) or "ring"
GOSSIP_FREQUENCY = 3      # Clients gossip (exchange models) every N rounds


# ── Model Definition ──────────────────────────────────────────────────────────
class MNISTNet(nn.Module):
    """
    Small CNN for MNIST classification.
    Architecture:
        Conv(1→32, 3x3) → ReLU
        Conv(32→64, 3x3) → ReLU
        MaxPool(2x2)          # 28x28 → 24x24 → 12x12
        Flatten               # 64 * 12 * 12 = 9216
        FC(9216→128) → ReLU
        FC(128→10)            # logits for 10 digit classes

    Note: fc2 is the "head" — in D-FedPer, this layer is kept personal
    and NOT shared during gossip.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)    # 28x28 → 26x26
        self.conv2 = nn.Conv2d(32, 64, 3)   # 26x26 → 24x24
        self.pool  = nn.MaxPool2d(2)         # 24x24 → 12x12
        self.fc1   = nn.Linear(9216, 128)    # 64*12*12 = 9216
        self.fc2   = nn.Linear(128, 10)      # classification head

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)             # flatten all dims except batch
        x = torch.relu(self.fc1(x))
        return self.fc2(x)                  # raw logits (no softmax — handled by CrossEntropyLoss)


# ── Data Preparation ──────────────────────────────────────────────────────────
# Standard MNIST normalization: mean=0.1307, std=0.3081 (computed over full training set)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    './data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(
    './data', train=False, download=True, transform=transform)

# Partition training data evenly across clients (IID split)
# Client i gets indices [i*data_per_client, (i+1)*data_per_client)
data_per_client = len(train_dataset) // NUM_CLIENTS
client_loaders = []

for i in range(NUM_CLIENTS):
    subset = torch.utils.data.Subset(
        train_dataset,
        list(range(i * data_per_client, (i + 1) * data_per_client))
    )
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    client_loaders.append(loader)

# Shared test loader — used to evaluate all client models
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=False,
    num_workers=0
)


# ── Topology Builder ──────────────────────────────────────────────────────────
def create_topology(num_clients, topology_type="mesh"):
    """
    Build a peer-to-peer communication graph as an adjacency list.

    Args:
        num_clients   : total number of nodes
        topology_type : "mesh" → every client connects to all others (fully connected)
                        "ring" → each client connects to its two immediate neighbors

    Returns:
        dict mapping client_id → list of neighbor_ids

    Examples (5 clients):
        mesh: {0:[1,2,3,4], 1:[0,2,3,4], ...}
        ring: {0:[1,4], 1:[2,0], 2:[3,1], ...}
    """
    if topology_type == "mesh":
        graph = {i: [j for j in range(num_clients) if j != i] for i in range(num_clients)}
    elif topology_type == "ring":
        graph = {i: [(i + 1) % num_clients, (i - 1) % num_clients] for i in range(num_clients)}
    return graph

topology = create_topology(NUM_CLIENTS, TOPOLOGY)


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_model(model):
    """
    Evaluate a single model on the global test set.

    Returns:
        Accuracy as a percentage (0–100).
    """
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
    return 100 * correct / len(test_dataset)


# ── Gossip Helper ─────────────────────────────────────────────────────────────
def gossip_average(client_models, topology):
    """
    Synchronous gossip: each client averages its own model with deep-copied
    snapshots from all its neighbors.

    We snapshot ALL models BEFORE any updates to avoid order-dependency
    (i.e., client 0's update should not affect client 1's input in same round).

    Args:
        client_models : list of nn.Module, one per client
        topology      : adjacency list dict

    Returns:
        None — modifies client_models in-place.
    """
    # Step 1: Snapshot every client's current state
    snapshots = [copy.deepcopy(m.state_dict()) for m in client_models]

    # Step 2: Each client averages itself with its neighbors' snapshots
    for client_id, model in enumerate(client_models):
        neighbor_ids   = topology[client_id]
        all_states     = [snapshots[client_id]] + [snapshots[nid] for nid in neighbor_ids]

        avg_state = {}
        for key in all_states[0].keys():
            # Stack tensors along a new dim and take the mean across neighbors
            avg_state[key] = torch.stack([s[key].float() for s in all_states]).mean(0)

        model.load_state_dict(avg_state)


# ═══════════════════════════════════════════════════════════════════════════════
# Algorithm 1: D-FedAvg
# ───────────────────────────────────────────────────────────────────────────────
# Standard federated averaging adapted for decentralized settings.
# Each client trains locally for LOCAL_EPOCHS, then periodically gossips
# its model weights with all neighbors and averages them.
# ═══════════════════════════════════════════════════════════════════════════════
def train_dfedavg():
    print("\n" + "=" * 60)
    print("🚀 D-FEDAVG")
    print("=" * 60)

    # Each client starts with its own randomly initialized model
    client_models = [MNISTNet().to(device) for _ in range(NUM_CLIENTS)]
    accuracies = []

    for round_num in tqdm(range(1, GLOBAL_ROUNDS + 1), desc="D-FedAvg"):

        # ── Local Training ────────────────────────────────────────────────────
        for client_id, model in enumerate(client_models):
            model.train()
            # Optimizer is re-created each round (no persistent momentum state)
            optimizer = optim.SGD(
                model.parameters(),
                lr=LR,
                momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY
            )

            for epoch in range(LOCAL_EPOCHS):
                for x, y in client_loaders[client_id]:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = nn.CrossEntropyLoss()(model(x), y)
                    loss.backward()
                    optimizer.step()

        # ── Gossip Aggregation ────────────────────────────────────────────────
        # Only gossip every GOSSIP_FREQUENCY rounds to reduce communication cost
        if round_num % GOSSIP_FREQUENCY == 0:
            gossip_average(client_models, topology)

        # ── Evaluation ───────────────────────────────────────────────────────
        # Report average accuracy across all client models
        avg_acc = np.mean([evaluate_model(m) for m in client_models])
        accuracies.append(avg_acc)

        if round_num % 5 == 0:
            print(f"  Round {round_num}: {avg_acc:.2f}%")

    print(f"✅ Final: {accuracies[-1]:.2f}%")
    return accuracies


# ═══════════════════════════════════════════════════════════════════════════════
# Algorithm 2: D-FedProx
# ───────────────────────────────────────────────────────────────────────────────
# Adds a proximal regularization term to the local loss to prevent client
# models from drifting too far from a reference (consensus) point.
#
# Loss_i = CE(model_i) + (mu/2) * ||w_i - w_ref||^2
#
# w_ref is computed as the mean of ALL client parameters at the start of
# each round. Note: in a truly decentralized setting, this would be
# approximated using only neighbor information.
# ═══════════════════════════════════════════════════════════════════════════════
def train_dfedprox():
    print("\n" + "=" * 60)
    print("🚀 D-FEDPROX")
    print("=" * 60)

    client_models = [MNISTNet().to(device) for _ in range(NUM_CLIENTS)]
    accuracies = []

    for round_num in tqdm(range(1, GLOBAL_ROUNDS + 1), desc="D-FedProx"):

        # ── Compute Reference (Consensus) Point ──────────────────────────────
        # Average all client parameters → used as the proximal anchor w_ref
        # NOTE: This requires global knowledge; in strict DFL this would be
        # replaced by a local neighborhood average.
        global_params = {}
        for key in client_models[0].state_dict().keys():
            global_params[key] = torch.stack([
                m.state_dict()[key].float() for m in client_models
            ]).mean(0).to(device)

        # ── Local Training with Proximal Term ────────────────────────────────
        for client_id, model in enumerate(client_models):
            model.train()
            optimizer = optim.SGD(model.parameters(), lr=LR)

            for epoch in range(LOCAL_EPOCHS):
                for x, y in client_loaders[client_id]:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()

                    # Standard task loss
                    loss = nn.CrossEntropyLoss()(model(x), y)

                    # Proximal term: penalizes distance from global reference
                    # Encourages all clients to stay near the consensus model
                    prox_term = 0
                    for name, param in model.named_parameters():
                        prox_term += ((param - global_params[name]) ** 2).sum()

                    total_loss = loss + (MU / 2) * prox_term
                    total_loss.backward()
                    optimizer.step()

        # ── Gossip Aggregation ────────────────────────────────────────────────
        if round_num % GOSSIP_FREQUENCY == 0:
            gossip_average(client_models, topology)

        # ── Evaluation ───────────────────────────────────────────────────────
        avg_acc = np.mean([evaluate_model(m) for m in client_models])
        accuracies.append(avg_acc)

        if round_num % 5 == 0:
            print(f"  Round {round_num}: {avg_acc:.2f}%")

    print(f"✅ Final: {accuracies[-1]:.2f}%")
    return accuracies


# ═══════════════════════════════════════════════════════════════════════════════
# Algorithm 3: D-FedPer (Personalized Federated Learning)
# ───────────────────────────────────────────────────────────────────────────────
# Splits the model into two parts:
#   - Shared body  : conv1, conv2, pool, fc1 — aggregated via gossip
#   - Personal head: fc2                     — kept local, never shared
#
# This allows clients to learn a shared feature representation while
# maintaining personalized classifiers suited to their local data distribution.
# ═══════════════════════════════════════════════════════════════════════════════
def train_dfedper():
    print("\n" + "=" * 60)
    print("🚀 D-FEDPER")
    print("=" * 60)

    client_models  = [MNISTNet().to(device) for _ in range(NUM_CLIENTS)]
    accuracies     = []

    for round_num in tqdm(range(1, GLOBAL_ROUNDS + 1), desc="D-FedPer"):

        # ── Local Training (full model including personal head) ───────────────
        for client_id, model in enumerate(client_models):
            model.train()
            optimizer = optim.SGD(model.parameters(), lr=LR)

            for epoch in range(LOCAL_EPOCHS):
                for x, y in client_loaders[client_id]:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = nn.CrossEntropyLoss()(model(x), y)
                    loss.backward()
                    optimizer.step()

        # ── Gossip (shared body only, fc2 is excluded) ───────────────────────
        if round_num % GOSSIP_FREQUENCY == 0:
            # Save each client's personal head BEFORE gossip
            personal_heads = [copy.deepcopy(m.fc2.state_dict()) for m in client_models]

            # Snapshot all models for synchronous gossip
            snapshots = [copy.deepcopy(m.state_dict()) for m in client_models]

            for client_id, model in enumerate(client_models):
                neighbor_ids = topology[client_id]
                all_states   = [snapshots[client_id]] + [snapshots[nid] for nid in neighbor_ids]

                avg_state = {}
                for key in all_states[0].keys():
                    if "fc2" in key:
                        # Personal head: keep this client's own weights unchanged
                        avg_state[key] = snapshots[client_id][key]
                    else:
                        # Shared body: average across self + neighbors
                        avg_state[key] = torch.stack([s[key].float() for s in all_states]).mean(0)

                model.load_state_dict(avg_state)

                # Explicitly restore personal head to guarantee it wasn't modified
                model.fc2.load_state_dict(personal_heads[client_id])

        # ── Evaluation ───────────────────────────────────────────────────────
        # Note: averaging accuracy across clients is a rough metric here —
        # each client has a personalized head, so per-client accuracy is
        # more informative in a real heterogeneous setting.
        avg_acc = np.mean([evaluate_model(m) for m in client_models])
        accuracies.append(avg_acc)

        if round_num % 5 == 0:
            print(f"  Round {round_num}: {avg_acc:.2f}%")

    print(f"✅ Final: {accuracies[-1]:.2f}%")
    return accuracies


# ═══════════════════════════════════════════════════════════════════════════════
# Algorithm 4: D-FedMask
# ───────────────────────────────────────────────────────────────────────────────
# Each client is assigned a binary sparse mask at initialization.
# During local training, masked (zeroed) weights receive no gradient updates —
# effectively each client trains a different sparse subnetwork.
#
# Gossip still averages full weight vectors. Ideally, aggregation would
# respect masks (only average weights active in both clients), but here
# a simple mean is used across all weights for simplicity.
# ═══════════════════════════════════════════════════════════════════════════════
def train_dfedmask():
    print("\n" + "=" * 60)
    print("🚀 D-FEDMASK")
    print("=" * 60)

    client_models = [MNISTNet().to(device) for _ in range(NUM_CLIENTS)]

    # ── Mask Initialization ───────────────────────────────────────────────────
    # For each client, create a binary mask per weight tensor.
    # MASK_SPARSITY fraction of weights are zeroed out (mask=0).
    # Bias parameters are always fully active (mask=1).
    masks = []
    for _ in range(NUM_CLIENTS):
        mask = {}
        for name, param in client_models[0].named_parameters():
            if "weight" in name:
                m = torch.ones_like(param)
                num_zero = int(param.numel() * MASK_SPARSITY)
                # Randomly select which weights to zero out
                zero_indices = torch.randperm(param.numel())[:num_zero]
                m.view(-1)[zero_indices] = 0
                mask[name] = m.to(device)
            else:
                # Biases are not masked
                mask[name] = torch.ones_like(param).to(device)
        masks.append(mask)

    accuracies = []

    for round_num in tqdm(range(1, GLOBAL_ROUNDS + 1), desc="D-FedMask"):

        # ── Local Training with Masked Gradients ─────────────────────────────
        for client_id, model in enumerate(client_models):
            model.train()
            optimizer = optim.SGD(model.parameters(), lr=LR)

            for epoch in range(LOCAL_EPOCHS):
                for x, y in client_loaders[client_id]:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = nn.CrossEntropyLoss()(model(x), y)
                    loss.backward()

                    # Zero out gradients for masked weights so they stay fixed
                    # This enforces sparse training — only active weights update
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad *= masks[client_id][name]

                    optimizer.step()

        # ── Gossip Aggregation ────────────────────────────────────────────────
        # Simple mean over all weights including masked ones.
        # A more principled approach would only average weights where
        # both clients have mask=1 (intersection masking).
        if round_num % GOSSIP_FREQUENCY == 0:
            gossip_average(client_models, topology)

        # ── Evaluation ───────────────────────────────────────────────────────
        avg_acc = np.mean([evaluate_model(m) for m in client_models])
        accuracies.append(avg_acc)

        if round_num % 5 == 0:
            print(f"  Round {round_num}: {avg_acc:.2f}%")

    print(f"✅ Final: {accuracies[-1]:.2f}%")
    return accuracies


# ═══════════════════════════════════════════════════════════════════════════════
# Main: Run all algorithms and plot results
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DECENTRALIZED FEDERATED LEARNING")
print("=" * 60)
print(f"Clients: {NUM_CLIENTS}, Topology: {TOPOLOGY}")
print(f"Local Epochs: {LOCAL_EPOCHS}, Gossip Every: {GOSSIP_FREQUENCY} rounds")
print("=" * 60)

# Run all four algorithms sequentially and collect per-round accuracies
results = {}
results["D-FedAvg"]  = train_dfedavg()
results["D-FedProx"] = train_dfedprox()
results["D-FedPer"]  = train_dfedper()
results["D-FedMask"] = train_dfedmask()

# ── Plotting ──────────────────────────────────────────────────────────────────
plt.figure(figsize=(14, 5))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Left: Accuracy curves over rounds
plt.subplot(1, 2, 1)
for (name, acc), color in zip(results.items(), colors):
    plt.plot(acc, marker='o', label=name, linewidth=2, markersize=4, color=color)
plt.axhline(95, linestyle="--", color='red', alpha=0.4, label='95% target')
plt.xlabel("Rounds")
plt.ylabel("Accuracy (%)")
plt.title("Decentralized FL Performance")
plt.legend()
plt.grid(alpha=0.3)
plt.ylim([0, 100])

# Right: Final accuracy bar chart
plt.subplot(1, 2, 2)
final_accs = [acc[-1] for acc in results.values()]
bars = plt.bar(results.keys(), final_accs, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel("Final Accuracy (%)")
plt.title("Final Performance Comparison")
plt.ylim([0, 100])
plt.axhline(95, linestyle="--", color='red', alpha=0.4)
for bar, acc in zip(bars, final_accs):
    plt.text(
        bar.get_x() + bar.get_width() / 2.,
        bar.get_height() + 1,
        f'{acc:.1f}%',
        ha='center', fontsize=10, fontweight='bold'
    )
plt.xticks(rotation=15)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("📊 FINAL RESULTS")
print("=" * 60)
for name, acc in results.items():
    print(f"{name:15s}: {acc[-1]:5.2f}%")
print("=" * 60)
