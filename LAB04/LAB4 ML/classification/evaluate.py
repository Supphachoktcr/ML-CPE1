import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from knn_tf import KNNClassifierModel

def evaluate_models(k_values, train_test_data):
    X_train, X_test, y_train, y_test = train_test_data
    results = {}
    best_k = None
    best_acc = 0.0

    os.makedirs('outputs', exist_ok=True)

    for k in k_values:
        model = KNNClassifierModel(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        results[k] = acc
        print(f"k = {k} -> Accuracy: {acc * 100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_k = k

    # 1. กราฟ k-curve
    plt.figure(figsize=(8, 5))
    plt.plot(list(results.keys()), list(results.values()), marker='o', color='b')
    plt.title('K Value vs Accuracy')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Test Accuracy')
    plt.grid(True)
    plt.xticks(k_values)
    plt.savefig('outputs/01_k_curve.png')
    plt.close()

    # 2. Confusion Matrix
    best_model = KNNClassifierModel(n_neighbors=best_k)
    best_model.fit(X_train, y_train)
    best_preds = best_model.predict(X_test)

    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Best k = {best_k})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('outputs/02_confusion_matrix.png')
    plt.close()

    return best_k, best_acc, best_preds