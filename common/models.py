import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def make_autoencoder(in_dim: int, bottleneck: int) -> nn.Module:
    """Symmetric autoencoder with a parameterised bottleneck width.

    Architecture: in_dim → 2·b → 4·b → b → 4·b → 2·b → in_dim
    where b = bottleneck.
    """
    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim,         2 * bottleneck), nn.ReLU(),
                nn.Linear(2 * bottleneck, 4 * bottleneck), nn.ReLU(),
                nn.Linear(4 * bottleneck, bottleneck),     nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck,     4 * bottleneck), nn.ReLU(),
                nn.Linear(4 * bottleneck, 2 * bottleneck), nn.ReLU(),
                nn.Linear(2 * bottleneck, in_dim),
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))
    return AE()


def train_autoencoder(X_tr, X_va, config: dict, verbose: bool = True):
    """Train one AE configuration. Returns (model, history, metrics_dict)."""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model     = make_autoencoder(X_tr.shape[1], config["bottleneck"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn   = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr)),
        batch_size=config["batch_size"], shuffle=True
    )
    X_va_t = torch.from_numpy(X_va).to(device)

    hist = {"train": [], "valid": []}
    best_val, best_state = float("inf"), None
    patience_counter = 0

    for epoch in range(config["epochs"]):
        model.train()
        running = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()
            running += loss.item() * batch.size(0)
        tr_loss = running / len(loader.dataset)

        model.eval()
        with torch.no_grad():
            va_loss = loss_fn(model(X_va_t), X_va_t).item()

        hist["train"].append(tr_loss)
        hist["valid"].append(va_loss)

        if va_loss < best_val:
            best_val, best_state = va_loss, {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.get("patience", 15):
                if verbose: print(f"  early stop at epoch {epoch}")
                break

        if verbose and (epoch % 10 == 0 or epoch == config["epochs"] - 1):
            print(f"  epoch {epoch:3d}  train={tr_loss:.5f}  valid={va_loss:.5f}")

    model.load_state_dict(best_state)

    return model, hist, {
        "bottleneck":    config["bottleneck"],
        "lr":            config["lr"],
        "batch_size":    config["batch_size"],
        "epochs_run":    len(hist["train"]),
        "best_val":      best_val,
        "final_train":   hist["train"][-1],
        "train_val_gap": hist["train"][-1] - best_val,
    }