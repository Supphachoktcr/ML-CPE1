from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    # ปรับ Path ให้ตรงกับที่ระบบบันทึกจริง
    weights = ROOT / 'runs' / 'detect' / 'runs' / 'train' / 'weights' / 'best.pt'

    if not weights.exists():
        raise FileNotFoundError(f'ไม่พบไฟล์โมเดลที่: {weights}')

    model = YOLO(str(weights))

    # ประเมินผลโมเดล
    metrics = model.val(
        data=str(ROOT / 'dataset' / 'data.yaml'),
        split='val',
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        plots=True,
        project=str(ROOT / 'runs'),
        name='evaluation_results',
        exist_ok=True,
    )

    print('\n===== Evaluation Results =====')
    print(f'Precision: {metrics.box.mp:.4f}')
    print(f'Recall: {metrics.box.mr:.4f}')
    print(f'mAP50: {metrics.box.map50:.4f}')
    print(f'mAP50-95: {metrics.box.map:.4f}')


if __name__ == '__main__':
    main()