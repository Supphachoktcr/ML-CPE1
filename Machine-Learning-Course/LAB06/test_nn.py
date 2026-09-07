"""Test the trained model on 4 random images (2x2 grid).

Random sample every run. Run main.py first.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

N_SAMPLES = 4


def test_nn(n_samples=N_SAMPLES):

    # Load model and test set
    model = joblib.load(f"{OUTPUT_DIR}/nn_model.pkl")
    X_test = np.load(f"{OUTPUT_DIR}/X_test.npy")
    y_test = np.load(f"{OUTPUT_DIR}/y_test.npy")
    with open(f"{OUTPUT_DIR}/classes.json") as f:
        classes = json.load(f)

    # Pick random images (no seed -> different every run)
    index = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[index]
    y_sample = y_test[index]

    # Flatten for scikit-learn model & scale 0-1
    X_sample_flat = X_sample.reshape(X_sample.shape[0], -1) / 255.0

    # Predict
    predictions = model.predict(X_sample_flat)

    # Get probabilities for confidence score
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_sample_flat)
        confidence = probabilities.max(axis=1)
    else:
        confidence = np.ones(n_samples)

    # Show results in a 2x2 grid
    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 4.0 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        if i >= n_samples:
            ax.axis("off")
            continue

        pred = classes[predictions[i]]
        true = classes[y_sample[i]]
        correct = predictions[i] == y_sample[i]
        color = "green" if correct else "red"

        ax.imshow(X_sample[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"Pred: {pred} ({confidence[i] * 100:.0f}%)\n" f"True: {true}",
            color=color,
        )

        print(
            f"[{i + 1}] Pred: {pred:<6} True: {true:<6} "
            f"conf {confidence[i] * 100:5.1f}%  "
            f"{'OK' if correct else 'WRONG'}"
        )

    correct_total = int((predictions == y_sample).sum())
    print(f"\nCorrect: {correct_total}/{n_samples}")

    fig.suptitle(f"Prediction: {correct_total}/{n_samples} correct")
    fig.tight_layout()

    save_path = f"{OUTPUT_DIR}/prediction_sample.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    test_nn()