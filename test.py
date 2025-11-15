import torch
from model import Evaluator

def debug_compare(model_path="best_model.pt", dataset_path="dataset.pt", n=10):
    # Load dataset
    data = torch.load(dataset_path)
    X = data["X"].float()
    y = data["y"].float()   # raw centipawns (before clamping/scaling)

    # Load model
    model = Evaluator(in_channels=X.shape[1], channels=32, n_blocks=4)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print("\n=== MODEL vs TARGET (10 samples) ===\n")

    for i in range(n):
        idx = torch.randint(0, len(X), (1,)).item()

        x_in = X[idx].unsqueeze(0)  # [1, C, 8, 8]

        with torch.no_grad():
            pred_scaled = model(x_in).item()       # in [-1, 1]
            pred_cp = pred_scaled * 100            # back to centipawns

        target_cp = y[idx].item()

        print(f"Index {idx:5d} |  Target: {target_cp:8.2f} cp  |  Predicted: {pred_cp:8.2f} cp")

    print("\nDone.\n")

if __name__ == "__main__":
    debug_compare("best_model.pt", "dataset.pt")
