"""Evaluate on the held-out test set.  Usage: python evaluate.py --data DATASET"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

from common import EMOTIONS, eval_tf, get_device, load_model

p = argparse.ArgumentParser()
p.add_argument("--data", default="DATASET")
p.add_argument("--model", default="models/best_model.pth")
args = p.parse_args()

device = get_device()
model = load_model(args.model, device)
dl = DataLoader(datasets.ImageFolder(f"{args.data}/test", transform=eval_tf), batch_size=64)

y_true, y_pred = [], []
with torch.no_grad():
    for x, y in dl:
        x = x.to(device)
        y_pred += (model(x).softmax(1) + model(torch.flip(x, dims=[3])).softmax(1)).argmax(1).cpu().tolist()  # + mirror TTA
        y_true += y.tolist()

report = classification_report(y_true, y_pred, labels=range(7), target_names=EMOTIONS, digits=3, zero_division=0)
print(report)
open("assets/classification_report.txt", "w").write(report)

cm = confusion_matrix(y_true, y_pred, labels=range(7), normalize="true")
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(7), EMOTIONS, rotation=45, ha="right"); ax.set_yticks(range(7), EMOTIONS)
for i in range(7):
    for j in range(7):
        ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > 0.5 else "black")
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (row-normalised)")
fig.colorbar(im); plt.tight_layout(); plt.savefig("assets/confusion_matrix.png", dpi=120)
print(f"Test accuracy: {np.mean(np.array(y_true) == np.array(y_pred)):.4f}")
