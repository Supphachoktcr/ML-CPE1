from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_svm(X_train, y_train, kernel="rbf", pca_components=150):
    # Scaler + PCA ใน Pipeline เพื่อให้ Test Data ถูก Transform แบบเดียวกันเสมอ
    n_samples, n_features = X_train.shape
    max_components = min(pca_components, n_samples, n_features)

    scaler = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=max_components, whiten=True, random_state=42)),
    ])

    # Fit และ Transform ข้อมูล Train
    X_train_scaled = scaler.fit_transform(X_train)

    # สร้างโมเดล SVM โดยรับค่า kernel จาก parameter (linear, poly, rbf)
    model = SVC(
        kernel=kernel, 
        C=10, 
        gamma="scale", 
        cache_size=1000,
        random_state=42
    )

    # Train โมเดล
    model.fit(X_train_scaled, y_train)

    return model, scaler


def predict_svm(model, scaler, X_test):
    # Scale และทำ PCA Transform ข้อมูล Test แบบเดียวกับข้อมูล Train
    X_test_scaled = scaler.transform(X_test)
    
    # พยากรณ์ผลลัพธ์
    predictions = model.predict(X_test_scaled)

    return predictions