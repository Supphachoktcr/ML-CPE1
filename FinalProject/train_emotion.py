from ultralytics import YOLO

def main():
    model = YOLO('yolov8n-cls.pt')

    print("เริ่มกระบวนการเทรนโมเดลจำแนกอารมณ์ (4 คลาส)...")
    results = model.train(
        data='./emotion_dataset',   
        epochs=50,                  
        imgsz=48,                   
        batch=32,                   
        device='0',                 
        project='emotion_weights',  
        name='yolov8_fer_4classes', 
        patience=10,
        workers=0
    )

if __name__ == '__main__':
    main()