from pathlib import Path
from ultralytics import YOLO

def main():
    # Load pretrained YOLO model
    model = YOLO("yolov8n.pt")

    # Train model
    model.train(
        data="dataset/data.yaml",
        epochs=50,          # สามารถปรับเพิ่มรอบได้ตามต้องการ (เช่น 30-50 รอบ)
        imgsz=640,          # ปรับขนาดภาพเป็น 640 เพื่อความแม่นยำที่ดีขึ้น
        batch=8,
        device=0,
        workers=0,
        amp=True,
        cache=False,
        project="runs",
        name="train",       # เปลี่ยนชื่อโฟลเดอร์ผลลัพธ์ให้ตรงกับไฟล์ evaluate, image_test ที่เราแก้ไป
        exist_ok=True,
        save=True,
        plots=True
    )

    # Output directory
    save_dir = Path(model.trainer.save_dir)

    print("Training completed!")
    print(f"Best Model: {save_dir / 'weights/best.pt'}")
    print(f"Last Model: {save_dir / 'weights/last.pt'}")
    print(f"Training CSV: {save_dir / 'results.csv'}")


if __name__ == "__main__":
    main()