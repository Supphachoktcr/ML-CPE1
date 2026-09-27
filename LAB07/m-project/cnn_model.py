import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping  # 1. นำเข้า EarlyStopping
import matplotlib.pyplot as plt
import os

def train_model(X_train, y_train, X_val, y_val, num_classes, output_dir, epochs=30, batch_size=32):
    
    # สร้างสถาปัตยกรรม CNN พื้นฐาน
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        Dense(1, activation='sigmoid') if num_classes == 2 else Dense(num_classes, activation='softmax')
    ])

    loss_function = 'binary_crossentropy' if num_classes == 2 else 'sparse_categorical_crossentropy'
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=loss_function,
                  metrics=['accuracy'])

    # 2. กำหนดเงื่อนไข Early Stopping (ให้จับตาดูค่า val_loss ถ้าไม่ลดลงติดต่อกัน 5 รอบ ให้หยุดทันที)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # เทรนโมเดล (ใส่ callbacks เพิ่มเข้าไป)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping]  # 3. นำมาใส่ตรงนี้
    )

    return model, history

def predict_model(model, X_test):
    predictions = model.predict(X_test)
    if predictions.shape[1] == 1:
        return (predictions > 0.5).astype("int32").flatten()
    else:
        return predictions.argmax(axis=1)