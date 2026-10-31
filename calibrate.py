"""Check the full webcam pipeline (detect -> align -> crop -> model) on RAF-DB test images
and pick the crop that matches training best.  Usage: python calibrate.py --data DATASET"""
import argparse
import json
import random

import cv2
import numpy as np
from PIL import Image
from torchvision import datasets

from common import align_face, detect_faces, get_device, largest, load_model, predict_pil

p = argparse.ArgumentParser()
p.add_argument("--data", default="DATASET")
p.add_argument("--n", type=int, default=600, help="number of test images to use")
args = p.parse_args()

device = get_device()
model = load_model(device=device)
ds = datasets.ImageFolder(f"{args.data}/test")
random.seed(0)
samples = random.sample(ds.samples, min(args.n, len(ds.samples)))
MARGINS = [-0.25, -0.2, -0.15, -0.1, -0.05, 0.0]
SHIFTS = [-0.1, -0.05, 0.0, 0.05, 0.1]
combos = [(m, s) for m in MARGINS for s in SHIFTS]

direct, found, correct = 0, 0, {c: 0 for c in combos}
for i, (path, y) in enumerate(samples, 1):
    face = np.array(Image.open(path).convert("RGB"))
    direct += int(np.argmax(list(predict_pil(model, Image.fromarray(face), device).values())) == y)
    # simulate a webcam frame: face enlarged and placed in a bigger scene
    frame = cv2.copyMakeBorder(cv2.resize(face, (200, 200)), 140, 140, 220, 220, cv2.BORDER_CONSTANT, value=(120, 120, 120))
    faces = detect_faces(frame)
    if faces:
        found += 1
        f = largest(faces)
        for m, s in combos:
            probs = predict_pil(model, Image.fromarray(align_face(frame, f, margin=m, shift=s)), device)
            correct[(m, s)] += int(np.argmax(list(probs.values())) == y)
    if i % 100 == 0:
        print(f"  {i}/{len(samples)} images checked…")

n, acc = len(samples), {c: v / max(found, 1) for c, v in correct.items()}
print(f"\nImages: {n} | face detected: {found / n:.1%}")
print(f"Model on original RAF-DB crops: {direct / n:.1%}  (target)\n")
print("Webcam pipeline accuracy (rows = margin, columns = vertical shift)")
print("margin  " + "".join(f"{s:>8}" for s in SHIFTS))
for m in MARGINS:
    print(f"{m:>6}  " + "".join(f"{acc[(m, s)]:>8.1%}" for s in SHIFTS))
best = max(combos, key=acc.get)
json.dump({"margin": best[0], "shift": best[1]}, open("models/pipeline.json", "w"))
print(f"\n✓ Best: margin {best[0]}, shift {best[1]} → {acc[best]:.1%}. Saved to models/pipeline.json (used by app.py)")
gap = direct / n - acc[best]
print("Pipeline matches training well." if gap < 0.03 else f"⚠ Pipeline is {gap:.1%} below the model — tell Claude.")
