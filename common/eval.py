import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

def recon_errors(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X)
        return ((model(X_t) - X_t) ** 2).mean(dim=1).numpy()

def calibrate_threshold(val_err, percentile=95):
    return float(np.percentile(val_err, percentile))

def metrics_from_scores(y_true, scores, threshold):
    yp = (scores > threshold).astype(int)
    return {
        "auc":       float(roc_auc_score(y_true, scores)),
        "f1":        float(f1_score(y_true, yp, zero_division=0)),
        "recall":    float(recall_score(y_true, yp, zero_division=0)),
        "precision": float(precision_score(y_true, yp, zero_division=0)),
    }

def per_type_breakdown(df, label_col="attack", type_col="subcategory", pred_col="pred"):
    return (df[df[label_col] == 1]
            .groupby(type_col)
            .agg(n=(pred_col, "size"),
                 caught=(pred_col, "sum"),
                 mean_err=("error", "mean"))
            .assign(recall=lambda d: d["caught"] / d["n"])
            .sort_values("recall", ascending=False))