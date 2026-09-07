import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # ตัดคอลัมน์ชื่อสัตว์ออก
    if 'animal_name' in df.columns:
        df = df.drop(columns=['animal_name'])
        
    # แยก Features (X) และ Target (y)
    if 'class_type' in df.columns:
        X = df.drop(columns=['class_type'])
        y = df['class_type']
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    
    # Split Train/Test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler