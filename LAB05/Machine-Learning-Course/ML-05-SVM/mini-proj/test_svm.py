import json
import os

import joblib
import numpy as np

from data_load import load_data
from preprocess import to_features
from split_data import split_dataset
from svm_model import train_svm, predict_svm
from evaluate import evaluate_model

DATA_PATH = "animal"     
OUTPUT_DIR = "outputs"  
IMG_SIZE = 100
TEST_SIZE = 0.2
MAX_PER_CLASS = None      # อ่านรูปภาพทั้งหมดในโฟลเดอร์ animal


def main():

    print("=" * 60)
    print("SVM Image Recognition: Animal Dataset")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)

    np.save(f"{OUTPUT_DIR}/images.npy", images)
    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Total images : {len(images)}")
    print(f"Classes      : {classes}")

    print("\n[Step 2] Preprocessing images...")

    X = to_features(images)
    y = labels
    print(f"Feature shape: {X.shape}")

    print("\n[Step 3] Splitting dataset...")

    X_train, X_test, y_train, y_test = split_dataset(X, y, TEST_SIZE)

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    kernels = ["linear", "poly", "rbf"]
    accuracy_results = {}

    print("\n" + "=" * 60)
    print("[Step 4-6] Training & Evaluating 3 SVM Kernels...")
    print("=" * 60)

    for k in kernels:
        print(f"\n>>> Kernel: [{k.upper()}] <<<")

        model, scaler = train_svm(X_train, y_train, kernel=k)

        joblib.dump(model, f"{OUTPUT_DIR}/svm_model_{k}.pkl")
        joblib.dump(scaler, f"{OUTPUT_DIR}/scaler_{k}.pkl")

        predictions = predict_svm(model, scaler, X_test)

        cm_save_path = f"{OUTPUT_DIR}/confusion_matrix_{k}.png"
        acc = evaluate_model(y_test, predictions, classes, save_path=cm_save_path)
        
        accuracy_results[k] = acc

    print("\n" + "=" * 60)
    print("FINAL OUTPUT SUMMARY")
    print("=" * 60)
    print("Accuracy Scores for each SVM Kernel:")
    for k, acc in accuracy_results.items():
        print(f"  - Kernel [{k.upper()}]: {acc * 100:.2f}% (Score: {acc:.4f})")

    # เซฟสรุปผล Accuracy ลงไฟล์ JSON
    with open(f"{OUTPUT_DIR}/accuracy_summary.json", "w") as f:
        json.dump(accuracy_results, f, indent=4)

    print(f"\nAll models and confusion matrices saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
