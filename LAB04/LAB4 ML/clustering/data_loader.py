import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data_for_clustering(file_path):
    df = pd.read_csv(file_path)
    
    # เลือกเฉพาะคอลัมน์ตัวเลข
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if 'class_type' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['class_type'])
        
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_df)
    
    return df, scaled_features