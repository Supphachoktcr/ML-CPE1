from pathlib import Path
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    # ปรับ Path ให้ตรงกับที่ระบบบันทึกจริง
    weights = ROOT / 'runs' / 'detect' / 'runs' / 'train' / 'weights' / 'best.pt'

    if not weights.exists():
        raise FileNotFoundError(f'ไม่พบไฟล์โมเดลที่: {weights}')

    model = YOLO(str(weights))

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError('ไม่สามารถเปิดใช้งานเว็บแคมได้ (Camera Index 0)')

    print("\nกำลังเปิดกล้อง... กดปุ่ม 'q' บนคีย์บอร์ดเพื่อปิดหน้าต่าง")

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print("ไม่สามารถอ่านภาพจากเว็บแคมได้")
                break

            results = model.predict(
                source=frame,
                conf=0.25,
                imgsz=640,
                device=0,
                verbose=False
            )

            annotated_frame = results[0].plot()

            cv2.imshow(
                'YOLO Real-time Animal Detection',
                annotated_frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print('\nReal-time testing completed!')


if __name__ == '__main__':
    main()