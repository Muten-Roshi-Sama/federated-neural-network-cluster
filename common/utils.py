import pickle, random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORE_ROOT = PROJECT_ROOT / "datasets" / "processed"

def set_seed(seed=42):
    random.seed(seed)
    import numpy as np; np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

def save_pickle(obj, name, subdir="mendeley"):
    path = STORE_ROOT / subdir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"saved → {path}  ({path.stat().st_size / 1024**2:.2f} MB)")

def load_pickle(name, subdir="mendeley"):
    with open(STORE_ROOT / subdir / name, "rb") as f:
        return pickle.load(f)