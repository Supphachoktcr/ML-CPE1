import cv2
from ultralytics import YOLO

def main():

    model_path = 'runs/classify/emotion_weights/yolov8_fer_4classes/weights/best.pt'
    try:
        model = YOLO(model_path)
    except FileNotFoundError:
        print(f"หาไฟล์ไม่พบ: {model_path}")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(1)
    print("เปิดกล้องเว็บแคม... (คลิกที่หน้าต่างวิดีโอแล้วกด 'q' เพื่อออก)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ไม่สามารถอ่านภาพจากกล้องได้")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            face_crop = frame[y:y+h, x:x+w]

            results = model(face_crop, verbose=False)
            
            top_class_index = results[0].probs.top1
            confidence = results[0].probs.top1conf.item()
            emotion_label = results[0].names[top_class_index]

            text = f"{emotion_label.upper()} {confidence*100:.1f}%"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Real-time Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()