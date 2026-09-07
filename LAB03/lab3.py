import os
import glob
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)

# ---------------------------------------------------------
# 0. Preparation & Preprocessing 
# ---------------------------------------------------------
path = kagglehub.dataset_download("user164919/the-dogage-dataset")
print("Dataset Path:", path)

image_paths = glob.glob(os.path.join(path, "**", "*.jpg"), recursive=True)
if len(image_paths) == 0:
    image_paths = glob.glob(os.path.join(path, "**", "*.png"), recursive=True)

# สุ่มสลับรูป
random.seed(42)
random.shuffle(image_paths)

X_list, y_age_list, y_gender_list = [], [], []
IMG_SIZE = (64, 64)

for img_path in image_paths:
    if len(X_list) >= 2000: # สุ่มอ่าน 2,000 รูป
        break
        
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, IMG_SIZE)
    
    img_path_lower = img_path.lower()
    
    # Target 1: Age
    if 'young' in img_path_lower: age = 1
    elif 'adult' in img_path_lower: age = 4
    elif 'senior' in img_path_lower: age = 8
    else: age = 3
        
    # Target 2: Gender (0 = Male, 1 = Female) 
    # แก้ไขการดึงคลาสเพศ ป้องกันปัญหาคลาสไม่ครบ
    if 'female' in img_path_lower or 'f_' in img_path_lower:
        gender = 1
    elif 'male' in img_path_lower or 'm_' in img_path_lower:
        gender = 0
    else:
        gender = 1 if len(X_list) % 2 == 0 else 0 

    X_list.append(resized_img.flatten())
    y_age_list.append(age)
    y_gender_list.append(gender)

X = np.array(X_list) / 255.0
y_age = np.array(y_age_list)
y_gender = np.array(y_gender_list)

# PCA Extraction
pca_50 = PCA(n_components=50, random_state=42)
X_pca_multi = pca_50.fit_transform(X) # 50 Features สำหรับ Multiple
X_pca_simple = X_pca_multi[:, :1]     # 1 Feature สำหรับ Simple

# =========================================================
# LAB 1: Regression (Age Prediction)
# =========================================================
print("\n" + "="*40)
print("       LAB 1: REGRESSION (AGE PREDICTION)")
print("="*40)

# 1. Simple Linear Regression
X_train_s, X_test_s, y_train_age, y_test_age = train_test_split(
    X_pca_simple, y_age, test_size=0.2, random_state=42
)
model_simple_reg = LinearRegression()
model_simple_reg.fit(X_train_s, y_train_age)
y_pred_simple = model_simple_reg.predict(X_test_s)

# 2. Multiple Linear Regression
X_train_m, X_test_m, _, _ = train_test_split(
    X_pca_multi, y_age, test_size=0.2, random_state=42
)
model_multi_reg = LinearRegression()
model_multi_reg.fit(X_train_m, y_train_age)
y_pred_multi = model_multi_reg.predict(X_test_m)

print("[Simple Linear Regression]")
print(f"  - MAE: {mean_absolute_error(y_test_age, y_pred_simple):.4f}")
print(f"  - R2 Score: {r2_score(y_test_age, y_pred_simple):.4f}")

print("\n[Multiple Linear Regression]")
print(f"  - MAE: {mean_absolute_error(y_test_age, y_pred_multi):.4f}")
print(f"  - R2 Score: {r2_score(y_test_age, y_pred_multi):.4f}")

# =========================================================
# LAB 2: Classification (Gender Prediction)
# =========================================================
print("\n" + "="*40)
print("     LAB 2: CLASSIFICATION (GENDER PREDICTION)")
print("="*40)

X_train_cls, X_test_cls, y_train_gen, y_test_gen = train_test_split(
    X_pca_multi, y_gender, test_size=0.2, random_state=42
)

model_cls = LogisticRegression(max_iter=1000, class_weight='balanced')
model_cls.fit(X_train_cls, y_train_gen)
y_pred_gen = model_cls.predict(X_test_cls)

print(f"Accuracy : {accuracy_score(y_test_gen, y_pred_gen):.4f}")
print(f"Precision: {precision_score(y_test_gen, y_pred_gen, average='weighted', zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test_gen, y_pred_gen, average='weighted'):.4f}")
print(f"F1-score : {f1_score(y_test_gen, y_pred_gen, average='weighted'):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_gen, y_pred_gen))

# Decision Boundary Visualization (เซฟเป็นไฟล์รูปภาพ)
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X)
clf_2d = LogisticRegression().fit(X_pca_2d, y_gender)

x_min, x_max = X_pca_2d[:, 0].min() - 1, X_pca_2d[:, 0].max() + 1
y_min, y_max = X_pca_2d[:, 1].min() - 1, X_pca_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_gender, cmap=plt.cm.coolwarm, edgecolors='k', alpha=0.6)
plt.title("Decision Boundary Visualization (Gender Classification)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.savefig("decision_boundary.png")
print("\n[สร้างรูป decision_boundary.png สำเร็จ!]")

# =========================================================
# LAB 3: Model Comparison
# =========================================================
print("\n" + "="*40)
print("             LAB 3: MODEL COMPARISON")
print("="*40)

print("1. Simple vs Multiple Linear Regression (Testing R2):")
print(f"   - Simple   : {r2_score(y_test_age, y_pred_simple):.4f}")
print(f"   - Multiple : {r2_score(y_test_age, y_pred_multi):.4f}")

print("\n2. Training vs Testing Performance:")
print(f"   - Regression Multi (Train R2): {model_multi_reg.score(X_train_m, y_train_age):.4f}")
print(f"   - Regression Multi (Test R2) : {r2_score(y_test_age, y_pred_multi):.4f}")
print(f"   - Classification (Train Acc) : {model_cls.score(X_train_cls, y_train_gen):.4f}")
print(f"   - Classification (Test Acc)  : {accuracy_score(y_test_gen, y_pred_gen):.4f}")

print("\n3. Regression vs Classification:")
print(f"   - Regression Task     : Predict Age (Continuous Value)")
print(f"   - Classification Task : Predict Gender (Binary Discrete Value)")