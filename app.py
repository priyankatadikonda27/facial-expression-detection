"""Real-time facial expression detection web app.  Run: python app.py  → http://localhost:7860"""
import io
import os
import threading

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from PIL import Image
from starlette.concurrency import run_in_threadpool

from common import align_face, detect_faces, gradcam_png, load_model, predict_pil

model = load_model()
lock = threading.Lock()  # one model call at a time (Grad-CAM uses hooks)
app = FastAPI()
HERE = os.path.dirname(os.path.abspath(__file__))
MAX_FACES = 6


def analyse(data: bytes, cam: bool):
    rgb = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    faces = sorted(detect_faces(rgb), key=lambda f: f["box"][2] * f["box"][3], reverse=True)[:MAX_FACES]
    res = {"size": [rgb.shape[1], rgb.shape[0]], "faces": []}
    with lock:
        for i, f in enumerate(faces):
            crop = Image.fromarray(align_face(rgb, f))
            item = {"box": [int(v) for v in f["box"]], "probs": predict_pil(model, crop)}
            if cam and i == 0:  # explain the main (largest) face only
                item["cam_label"], item["cam"] = gradcam_png(model, crop)
            res["faces"].append(item)
    return res


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.post("/predict")
async def predict(request: Request, cam: int = 0):
    return await run_in_threadpool(analyse, await request.body(), bool(cam))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    if "SPACE_ID" not in os.environ:  # running locally (not on Hugging Face)
        import webbrowser
        url = f"http://localhost:{port}"
        print(f"\n  ➜  Open {url}\n")
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="0.0.0.0", port=port)
