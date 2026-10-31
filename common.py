"""Shared helpers: labels, transforms, model, face detection + alignment."""
import json
import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

# RAF-DB folder names 1..7 map to these labels (in order)
EMOTIONS = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"]
IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
HERE = os.path.dirname(os.path.abspath(__file__))

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1)),  # robust to crop size
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.ColorJitter(0.3, 0.3, 0.2),
    transforms.RandomGrayscale(0.1),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),  # webcam blur
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes=len(EMOTIONS), pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
    return m


def load_model(path=os.path.join(HERE, "models", "best_model.pth"), device="cpu"):
    ckpt = torch.load(path, map_location=device)
    m = build_model(pretrained=False)
    m.load_state_dict(ckpt["state_dict"])
    return m.to(device).eval()


@torch.no_grad()
def predict_pil(model, pil_img, device="cpu"):
    """Probabilities, averaged with the mirrored image (test-time augmentation)."""
    x = eval_tf(pil_img.convert("RGB")).unsqueeze(0).to(device)
    x = torch.cat([x, torch.flip(x, dims=[3])])
    probs = torch.softmax(model(x), dim=1).mean(0).cpu().tolist()
    return dict(zip(EMOTIONS, probs))


# ---------------- Face detection (YuNet, falls back to Haar) ----------------
YUNET = os.path.join(HERE, "models", "face_detection_yunet_2023mar.onnx")
_yunet = cv2.FaceDetectorYN.create(YUNET, "", (320, 320), 0.7) if os.path.exists(YUNET) else None
_haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

_cfg = os.path.join(HERE, "models", "pipeline.json")
_p = json.load(open(_cfg)) if os.path.exists(_cfg) else {}
MARGIN, SHIFT = _p.get("margin", -0.1), _p.get("shift", 0.0)  # set by calibrate.py


def detect_faces(rgb):
    """rgb: HxWx3 uint8 -> list of dicts {box:(x,y,w,h), eyes:((rx,ry),(lx,ly)) or None}."""
    if _yunet is not None:
        h, w = rgb.shape[:2]
        _yunet.setInputSize((w, h))
        _, faces = _yunet.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return [{"box": tuple(f[:4]), "eyes": ((f[4], f[5]), (f[6], f[7]))} for f in (faces if faces is not None else [])]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return [{"box": tuple(b), "eyes": None} for b in _haar.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))]


def align_face(rgb, face, margin=None, shift=None, out=IMG_SIZE):
    """Rotate so the eyes are level, then take a square crop around the face (like RAF-DB's aligned images)."""
    margin = MARGIN if margin is None else margin  # negative = tighter than the detector box
    shift = SHIFT if shift is None else shift      # move crop centre down (+) / up (-), fraction of box height
    x, y, w, h = face["box"]
    cx, cy = x + w / 2, y + h / 2 + shift * h
    angle = 0.0
    if face["eyes"] is not None:
        (rx, ry), (lx, ly) = face["eyes"]
        angle = math.degrees(math.atan2(ly - ry, lx - rx))
    side = max(w, h) * (1 + 2 * margin)
    M = cv2.getRotationMatrix2D((cx, cy), angle, out / side)
    M[:, 2] += (out / 2 - cx, out / 2 - cy)
    return cv2.warpAffine(rgb, M, (out, out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def largest(faces):
    return max(faces, key=lambda f: f["box"][2] * f["box"][3])


def gradcam_png(model, pil_img):
    """Grad-CAM: heatmap of the face regions that drove the top prediction. Returns (label, data-URL PNG)."""
    import base64
    x = eval_tf(pil_img.convert("RGB")).unsqueeze(0)
    store = {}
    hook = model.layer4.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    try:
        with torch.enable_grad():
            out = model(x)
            idx = int(out.argmax(1))
            grad = torch.autograd.grad(out[0, idx], store["act"])[0]
    finally:
        hook.remove()
    act = store["act"].detach()
    cam = torch.relu((grad.mean((2, 3), keepdim=True) * act).sum(1))[0].numpy()
    cam = cv2.resize(cam / (cam.max() + 1e-8), (IMG_SIZE, IMG_SIZE))
    face = np.array(pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    blend = cv2.addWeighted(face, 0.55, heat, 0.45, 0)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))
    return EMOTIONS[idx], "data:image/png;base64," + base64.b64encode(buf).decode()
