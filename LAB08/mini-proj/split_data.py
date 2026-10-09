import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(X, y, test_size=0.2, val_size=0.1, random_state=42):

    X = np.asarray(X)
    y = np.asarray(y)

    if test_size <= 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=random_state, stratify=y
        )
        return X_train, X_val, np.array([]), y_train, y_val, np.array([])

    # 1. Separate Test set first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 2. Calculate Validation set ratio from remaining data
    val_ratio = val_size / (1.0 - test_size)

    # 3. Separate Validation set from Training set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, random_state=random_state, stratify=y_train
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def split_train_to_test(X_train_raw, y_train_raw, test_size=0.15, random_state=42):
    """
    Splits a portion of the raw training dataset to serve as the test set.
    """
    X_train_raw = np.asarray(X_train_raw)
    y_train_raw = np.asarray(y_train_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_train_raw,
        y_train_raw,
        test_size=test_size,
        random_state=random_state,
        stratify=y_train_raw
    )

    return X_train, X_test, y_train, y_test