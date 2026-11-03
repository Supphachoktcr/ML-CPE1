import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # 1. โหลดโมเดลที่เทรนเสร็จแล้ว
    model_path = 'runs/classify/emotion_weights/yolov8_fer_4classes/weights/best.pt'
    model = YOLO(model_path)

    # 2. กำหนดพาทของโฟลเดอร์ test และรายชื่อคลาสอารมณ์
    test_dir = './emotion_dataset/test'
    class_names = sorted(os.listdir(test_dir))  # ดึงชื่อโฟลเดอร์ย่อยมาเป็นชื่อคลาส
    
    y_true = []  # เก็บคำตอบจริง
    y_pred = []  # เก็บคำตอบที่โมเดลทายได้

    print("กำลังประมวลผลรูปภาพในชุด Test Set... (อาจใช้เวลาสักครู่)")
    
    # 3. วนลูปอ่านรูปภาพในแต่ละโฟลเดอร์อารมณ์
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            
            # โหลดรูปภาพ
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # ให้โมเดลทำนาย
            results = model(img, verbose=False)
            
            # ดึง index ของคลาสที่โมเดลมั่นใจที่สุด
            pred_idx = results[0].probs.top1
            
            y_true.append(class_idx)
            y_pred.append(pred_idx)

    # 4. แสดงผล Classification Report (Precision, Recall, F1-Score)
    print("\n" + "="*50)
    print("รายงานผลการประเมิน (Classification Report):")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # 5. วาดกราฟ Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Emotion Recognition')
    plt.ylabel('Actual Emotion')
    plt.xlabel('Predicted Emotion')
    
    # บันทึกรูปกราฟและแสดงผล
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("บันทึกกราฟ Confusion Matrix สำเร็จ (confusion_matrix.png)")
    plt.show()

if __name__ == '__main__':
    main()