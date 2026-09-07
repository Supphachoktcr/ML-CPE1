import os
import joblib
import warnings
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.exceptions import ConvergenceWarning

# ปิด Warning เรื่อง ConvergenceWarning ไม่ให้รกหน้าจอ
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def train_model(X_train, y_train, X_val, y_val, num_classes, output_dir, epochs, batch_size):
    # 1. Flatten ภาพ
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    if X_train_flat.max() > 1.0:
        X_train_flat = X_train_flat / 255.0
        X_val_flat = X_val_flat / 255.0

    # 2. ปรับ Learning Rate ให้เล็กลง เพื่อหยุดอาการแกว่ง
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        learning_rate_init=0.0001,  # ลด Learning Rate ลง 10 เท่า
        alpha=0.001,
        max_iter=1,                 # รันทีละ 1 iter
        warm_start=True,
        random_state=42
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print("\nTraining Neural Network...")
    
    # 3. เทรนจริงทีละรอบโดยใช้ partial_fit แทน fit
    classes = np.unique(y_train)
    for epoch in range(epochs):
        model.partial_fit(X_train_flat, y_train, classes=classes) # ใช้ partial_fit จะไม่ reset momentum

        # คำนวณค่าเพื่อเอาไปพล็อต
        train_pred = model.predict(X_train_flat)
        val_pred = model.predict(X_val_flat)
        
        train_prob = model.predict_proba(X_train_flat)
        val_prob = model.predict_proba(X_val_flat)

        train_losses.append(log_loss(y_train, train_prob, labels=classes))
        val_losses.append(log_loss(y_val, val_prob, labels=classes))

        train_accs.append(accuracy_score(y_train, train_pred))
        val_accs.append(accuracy_score(y_val, val_pred))

    class History:
        def __init__(self, t_loss, v_loss, t_acc, v_acc):
            self.history = {
                'loss': t_loss,
                'val_loss': v_loss,
                'accuracy': t_acc,
                'val_accuracy': v_acc
            }

    history = History(train_losses, val_losses, train_accs, val_accs)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(model, os.path.join(output_dir, "nn_model.pkl"))

    return model, history
def predict_model(model, X_test):
    # Flatten ภาพก่อนนำไปทำนายผล
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # ป้องกันการหารซ้ำถ้า X_test โดน scale มาแล้ว
    if X_test_flat.max() > 1.0:
        X_test_flat = X_test_flat / 255.0
        
    return model.predict(X_test_flat)