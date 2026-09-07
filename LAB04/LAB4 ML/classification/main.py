import pandas as pd
from data_loader import load_and_preprocess_data
from evaluate import evaluate_models

def main():
    data_path = '../data-animal/animal_dataset.csv'
    
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(data_path)
    
    k_list = [3, 5, 7]
    best_k, best_acc, best_preds = evaluate_models(k_list, (X_train, X_test, y_train, y_test))
    
    pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': best_preds})
    pred_df.to_csv('outputs/predictions.csv', index=False)
    
    print("\n--- Classification Completed ---")
    print(f"Best k = {best_k} (Accuracy: {best_acc * 100:.2f}%)")

if __name__ == "__main__":
    main()