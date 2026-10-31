# Facial Expression Detection

Detects faces in a photo or webcam frame and classifies each into one of 7 emotions —
**Surprise, Fear, Disgust, Happy, Sad, Angry, Neutral** — using a ResNet18 fine-tuned on the
RAF-DB dataset. Comes with a Gradio web app.

> 🔗 **Live demo:** _add your Hugging Face Space link here_

![Confusion matrix](assets/confusion_matrix.png)

## How it works

```
Image ──► Haar cascade face detector ──► crop face ──► ResNet18 (ImageNet-pretrained,
          (OpenCV)                                       fine-tuned on RAF-DB) ──► 7 emotion probabilities
```

| Part | Choice | Why |
|---|---|---|
| Dataset | RAF-DB basic (~12.3k train / ~3.1k test, 100×100 aligned faces) | Real-world faces, varied pose/lighting |
| Model | ResNet18, transfer learning | Strong accuracy, small (~45 MB), trains on a laptop |
| Imbalance | Class-weighted cross-entropy + label smoothing | "Happy" is ~40% of the data; "Fear"/"Disgust" are rare |
| Augmentation | Flip, rotation, colour jitter, random erasing | Reduces overfitting |
| Validation | 10% of train held out; test set used **once** at the end | Honest test score |
| Face detection | OpenCV Haar cascade | Bundled with OpenCV, no extra download |

## Results

| Metric | Value |
|---|---|
| Test accuracy | 84.2% |
| Macro F1 | 0.759 |

Training curves: `assets/training_curves.png` · Full report: `assets/classification_report.txt`

## Project structure

```
facial-expression-detection/
├── common.py          # labels, transforms, model, face detection
├── train.py           # training loop → models/best_model.pth
├── evaluate.py        # test-set metrics + confusion matrix
├── predict.py         # CLI: python predict.py photo.jpg
├── app.py             # Gradio web app (upload + webcam)
├── requirements.txt
├── models/            # trained weights (best_model.pth)
└── assets/            # plots and reports for this README
```

## Run it locally

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# put the RAF-DB folder here as DATASET/train/1..7 and DATASET/test/1..7
python train.py --data DATASET --epochs 15
python evaluate.py --data DATASET
python app.py            # open http://127.0.0.1:7860
```

## Dataset & licence

RAF-DB is for **non-commercial research only** and may not be redistributed, so it is not
included in this repository. Get it from the official site or Kaggle and place it in `DATASET/`.
Label mapping: folders `1..7` = Surprise, Fear, Disgust, Happy, Sad, Angry, Neutral.

## Limitations

- Trained on posed/aligned face crops; accuracy drops on side profiles, low light, and small faces.
- Haar cascade misses some faces (tilted, occluded). MediaPipe or RetinaFace would be stronger.
- "Fear" and "Disgust" are the weakest classes because they have the fewest samples.
- Facial expression ≠ inner feeling. This model reads visible expressions only.

## Author

Priyanka Tadikonda · [GitHub](https://github.com/priyankatadikonda27) · [LinkedIn](https://linkedin.com/in/priyanka-tadikonda-017b05371)
