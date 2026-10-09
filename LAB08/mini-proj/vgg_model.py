import json
import os

from tensorflow import keras
from tensorflow.keras import layers

# Layout VGG16 ดั้งเดิม (13 conv layers: 2-2-3-3-3)
VGG16 = [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]

# Lighter variant สำหรับ CPU / Dataset ขนาดเล็ก
VGG_SMALL = [(32, 2), (64, 2), (128, 3), (256, 3)]


def vgg_block(model, filters, n_conv):
    """ Conv2D -> ReLU -> MaxPooling2D แบบ VGG Standard """
    for _ in range(n_conv):
        model.add(layers.Conv2D(filters, 3, padding="same", activation="relu"))

    model.add(layers.MaxPooling2D(2))


def build_model(input_shape, num_classes, blocks=VGG_SMALL):

    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))

    # แปลงภาพจาก 0-255 เป็น 0.0-1.0
    model.add(layers.Rescaling(1.0 / 255.0))

    # Data Augmentation พื้นฐาน
    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.1))

    # Feature Extractor
    for filters, n_conv in blocks:
        vgg_block(model, filters, n_conv)

    model.add(layers.GlobalAveragePooling2D())
    
    # Classifier Head
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.4))

    model.add(layers.Dense(
        1 if num_classes == 2 else num_classes,
        activation="sigmoid" if num_classes == 2 else "softmax"
    ))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy" if num_classes == 2
             else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(X_train, y_train, X_val, y_val, num_classes,
                output_dir=None, epochs=35, batch_size=32, blocks=VGG_SMALL):

    model = build_model(X_train.shape[1:], num_classes, blocks)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        model.save(os.path.join(output_dir, "vgg_model.keras"))
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump({k: [float(v) for v in vs]
                       for k, vs in history.history.items()}, f)

        print(f"Saved: {os.path.join(output_dir, 'vgg_model.keras')}")

    return model, history


def predict_model(model, X_test):

    probabilities = model.predict(X_test, verbose=0)

    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() > 0.5).astype(int)

    return probabilities.argmax(axis=1)