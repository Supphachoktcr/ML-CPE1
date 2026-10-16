# ML-10-Object Detection II

This project implements **YOLOv8-based object detection** using the Berkeley DeepDrive (BDD100K) dataset. It detects road objects such as vehicles, pedestrians, and traffic infrastructure, and covers training, evaluation, image inference, and real-time webcam detection.
 
## 1. Project Overview

The project fine-tunes **Ultralytics YOLOv8n** with pretrained weights on the BDD100K dataset.

**Features:**
- Detection across 10 road-object classes
- Pretrained YOLOv8 fine-tuning
- Evaluation using Precision, Recall, mAP50, and mAP50-95
- Image and real-time webcam inference
- Automatic model checkpoints and epoch-by-epoch CSV training logs

## 2. Dataset
https://www.kaggle.com/datasets/sc0v1n0/animal-picture-set
git clone Machine-Learning-Course/ML-06-NN at main · aproot-en/Machine-Learning-Course
## 3. Project Structure

```text
ML-10-Object-Detection-II/
├── dataset/
│   ├── train/{images,labels}/
│   ├── val/{images,labels}/
│   └── data.yaml
├── train.py
├── evaluate.py
├── image_test.py
├── realtime_test.py
├── requirements.txt
└── README.md
```

| Script | Purpose |
|---|---|
| `train.py` | Train YOLOv8 and save checkpoints and training logs |
| `evaluate.py` | Evaluate on labeled test images |
| `image_test.py` | Detect objects in randomly selected test images |
| `realtime_test.py` | Run webcam-based real-time detection |





