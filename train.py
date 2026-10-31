"""Fine-tune ResNet18 on RAF-DB.  Usage: python train.py --data DATASET --epochs 15"""
import argparse
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from common import EMOTIONS, build_model, eval_tf, get_device, train_tf

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="DATASET")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="models/best_model.pth")
    p.add_argument("--resume", default="", help="start from an existing checkpoint (fine-tuning)")
    args = p.parse_args()

    torch.manual_seed(42)
    device = get_device()
    print("Device:", device)

    # Two views of the same folder: augmented for training, clean for validation
    train_full = datasets.ImageFolder(f"{args.data}/train", transform=train_tf)
    val_full = datasets.ImageFolder(f"{args.data}/train", transform=eval_tf)
    assert train_full.classes == [str(i) for i in range(1, 8)], f"Unexpected folders: {train_full.classes}"

    idx = torch.randperm(len(train_full)).tolist()
    n_val = int(len(idx) * args.val_split)
    train_ds, val_ds = Subset(train_full, idx[n_val:]), Subset(val_full, idx[:n_val])
    train_dl = DataLoader(train_ds, args.bs, shuffle=True, num_workers=args.workers)
    val_dl = DataLoader(val_ds, args.bs, num_workers=args.workers)

    # Class weights to counter imbalance (Happy dominates RAF-DB)
    counts = Counter(train_full.targets[i] for i in idx[n_val:])
    print("Train counts:", {EMOTIONS[k]: v for k, v in sorted(counts.items())})
    # sqrt-dampened weights: helps rare classes without making the model over-predict them
    w = torch.tensor([(len(train_ds) / (7 * counts.get(c, 1))) ** 0.5 for c in range(7)], dtype=torch.float)

    model = build_model(pretrained=not (args.no_pretrained or args.resume))
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu")["state_dict"])
        print("Resumed from", args.resume)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)


    def run(dl, train):
        model.train(train)
        total, correct, loss_sum = 0, 0, 0.0
        with torch.set_grad_enabled(train):
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                loss_sum += loss.item() * len(y)
                correct += (out.argmax(1) == y).sum().item()
                total += len(y)
        return loss_sum / total, correct / total


    hist, best = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}, 0.0
    for ep in range(1, args.epochs + 1):
        tl, ta = run(train_dl, True)
        vl, va = run(val_dl, False)
        sched.step()
        for k, v in zip(hist, (tl, vl, ta, va)):
            hist[k].append(v)
        print(f"Epoch {ep:2d} | train loss {tl:.3f} acc {ta:.3f} | val loss {vl:.3f} acc {va:.3f}")
        if va > best:
            best = va
            torch.save({"state_dict": model.state_dict(), "classes": EMOTIONS}, args.out)
            print("  ✓ saved best model")

    json.dump(hist, open("models/history.json", "w"), indent=2)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k in ("train_loss", "val_loss"):
        ax[0].plot(hist[k], label=k)
    for k in ("train_acc", "val_acc"):
        ax[1].plot(hist[k], label=k)
    for a, t in zip(ax, ("Loss", "Accuracy")):
        a.set_title(t); a.set_xlabel("epoch"); a.legend()
    plt.tight_layout(); plt.savefig("assets/training_curves.png", dpi=120)
    print(f"Best val acc: {best:.4f}")


if __name__ == "__main__":
    main()
