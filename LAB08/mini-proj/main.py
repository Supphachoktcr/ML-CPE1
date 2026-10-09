import json
import os

import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_train_to_test
from vgg_model import train_model, predict_model, VGG16, VGG_SMALL
from evaluate import evaluate_model, plot_history

# BASE_DIR คือโฟลเดอร์ mini-proj
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ANIMALS_DIR ชี้ถอยออกไป 1 ชั้น (..) จะเจอกับโฟลเดอร์ Animals
ANIMALS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Animals"))
TRAIN_DIR = os.path.join(ANIMALS_DIR, "train")
VALID_DIR = os.path.join(ANIMALS_DIR, "valid")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE = 100
TEST_SIZE = 0.15  # แบ่ง 15% จาก Train Set เพื่อใช้เป็น Test Set สำหรับวัดผล
MAX_PER_CLASS = None  # None = โหลดรูปภาพทั้งหมด
EPOCHS = 35
BATCH_SIZE = 32

# ใช้ VGG_SMALL เพื่อป้องกัน Overfitting และประมวลผลบน CPU ได้เร็วขึ้น
BLOCKS = VGG_SMALL


def main():

    print("--" * 30)
    print("VGG (Deep CNN) Image Recognition: Animals Classifier")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")
    raw_images_train, raw_labels_train, classes = load_data(TRAIN_DIR, IMG_SIZE, MAX_PER_CLASS)
    raw_images_val, raw_labels_val, _ = load_data(VALID_DIR, IMG_SIZE, MAX_PER_CLASS)

    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Classes : {classes}")

    # Step 2: Preprocessing
    print("\n[Step 2] Preprocessing images...")
    X_train_full = to_features(raw_images_train)
    y_train_full = raw_labels_train

    X_val = to_features(raw_images_val)
    y_val = raw_labels_val

    # Step 3: Split Test set from Train set
    print("\n[Step 3] Splitting dataset...")
    X_train, X_test, y_train, y_test = split_train_to_test(
        X_train_full, y_train_full, test_size=TEST_SIZE
    )

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_val.npy", X_val)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_val.npy", y_val)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples   : {len(X_train)}")
    print(f"Validation samples : {len(X_val)}")
    print(f"Testing samples    : {len(X_test)}")

    # Step 4: Train Model
    print("\n[Step 4] Training model...")
    model, history = train_model(
        X_train, y_train, X_val, y_val, len(classes),
        OUTPUT_DIR, EPOCHS, BATCH_SIZE, BLOCKS
    )

    print("Training completed.")

    # Step 5: Prediction
    print("\n[Step 5] Testing model...")
    predictions = predict_model(model, X_test)

    # Step 6: Evaluation
    print("\n[Step 6] Evaluating model...")
    evaluate_model(y_test, predictions, classes,
                   save_path=f"{OUTPUT_DIR}/confusion_matrix.png")
    plot_history(history, f"{OUTPUT_DIR}/training_history.png")


if __name__ == "__main__":
    main()