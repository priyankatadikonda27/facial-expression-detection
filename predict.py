"""Predict emotions for faces in an image.  Usage: python predict.py path/to/photo.jpg"""
import sys

import numpy as np
from PIL import Image

from common import align_face, detect_faces, load_model, predict_pil

model = load_model()
rgb = np.array(Image.open(sys.argv[1]).convert("RGB"))
faces = detect_faces(rgb)
if not faces:
    print("No face found.")
for i, f in enumerate(faces, 1):
    probs = predict_pil(model, Image.fromarray(align_face(rgb, f)))
    top = max(probs, key=probs.get)
    print(f"Face {i} at {tuple(int(v) for v in f['box'])}: {top} ({probs[top]:.1%})")
