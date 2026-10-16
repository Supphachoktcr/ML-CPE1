from pathlib import Path
import random
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    # Path ของโมเดลที่เทรนเสร็จแล้ว
    weights = ROOT / 'runs' / 'detect' / 'runs' / 'train' / 'weights' / 'best.pt'
    
    # เปลี่ยนมาใช้โฟลเดอร์ valid แทน test เพราะชุดข้อมูลเรามี train กับ valid
    image_dir = ROOT / 'dataset' / 'valid' / 'images'

    if not weights.exists():
        raise FileNotFoundError(f'ไม่พบไฟล์โมเดลที่: {weights}')

    if not image_dir.exists():
        raise FileNotFoundError(f'ไม่พบโฟลเดอร์ Dataset: {image_dir}')

    model = YOLO(str(weights))

    extensions = {'.jpg', '.jpeg', '.png'}
    images = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in extensions
    ]

    if not images:
        raise RuntimeError('ไม่พบรูปภาพในโฟลเดอร์ valid')

    # สุ่มเลือกรูปภาพมาทดสอบ 5 รูป
    selected_images = random.sample(
        images, min(5, len(images))
    )

    for image in selected_images:
        results = model.predict(
            source=str(image),
            conf=0.1,
            imgsz=640,
            device=0,
            save=True,
            project=str(ROOT / 'runs'),
            name='image_test',
            exist_ok=True
        )

        for result in results:
            print(f'\nImage: {image.name}')

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                print(
                    f'{model.names[class_id]}: '
                    f'{confidence:.2f}'
                )

            annotated = result.plot()
            cv2.imshow('YOLO Image Detection', annotated)

            # กดปุ่ม 'q' เพื่อปิด หรือกดปุ่มใดๆ เพื่อดูรูปถัดไป
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print('\nImage testing completed!')


if __name__ == '__main__':
    main()