import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import random

from model import Evaluator   # your model class


def debug_samples(model, dataset, device, n=10):
    print("\n=== Sample Predictions ===")
    model.eval()

    for _ in range(n):
        idx = random.randint(0, len(dataset) - 1)

        X = dataset.X[idx].unsqueeze(0).to(device)
        y_cp_raw = dataset.y[idx].item() * 600.0        # scaled → cp
        with torch.no_grad():
            pred_scaled = model(X).item()
            pred_cp = pred_scaled * 600.0

        print(f"Idx {idx:5d} | Target: {y_cp_raw:8.2f} cp | Pred: {pred_cp:8.2f} cp")

    print("===========================\n")

# ============================================================
# 1. Dataset: Load → Clamp → Scale
# ============================================================
class ChessEvalDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path)

        # Input planes
        self.X = data["X"].float()

        # Raw engine evals in centipawns (already side-to-move perspective)
        y = data["y"].float()

        # Clamp to safe range
        y = torch.clamp(y, -600, 600)

        # Scale to [-6, 6]
        y = y / 600.0

        self.y = y.unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# 2. One Epoch of Training
# ============================================================
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for X, y in tqdm(loader, desc="Training", leave=False):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(X)

        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


# ============================================================
# 3. Validation
# ============================================================
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)

            loss = loss_fn(preds, y)
            total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


# ============================================================
# 4. Main Training Loop
# ============================================================
def train_model(
    dataset_path="dataset.pt",
    epochs=30,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=1e-5,
):
    # Device (MPS on Mac, else CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # Dataset
    dataset = ChessEvalDataset(dataset_path)

    # Train/val split
    train_size = int(0.9 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)

    # Model
    in_channels = dataset.X.shape[1]  # e.g., 19 planes
    model = Evaluator(
        in_channels=in_channels,
        channels=32,
        n_blocks=8
    ).to(device)

    print(model)
    print("\nTotal parameters:", sum(p.numel() for p in model.parameters()))

    # Loss & Optimizer
    # loss_fn = nn.SmoothL1Loss(beta=0.1)
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3
    )

    best_val = float("inf")
    save_path = "best_model.pt"

    # Epoch Loop
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss   = validate(model, val_loader, loss_fn, device)

        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")

        # === PRINT 10 DEBUG SAMPLES EACH EPOCH ===
        debug_samples(model, dataset, device, n=10)

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path}")

    print("\nTraining complete.")
    print("Best validation loss:", best_val)


if __name__ == "__main__":
    train_model("dataset2.pt", epochs=25)
