import cv2
import numpy as np


def preprocess_image(image, img_size=100):
    """Resize one image to img_size x img_size RGB. None if unusable."""

    if image is None or image.size == 0:
        return None

    # cv2 reads BGR, convert to RGB so images display correctly
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image (INTER_AREA is the right filter for shrinking)
    image = cv2.resize(
        image,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA
    )

    # 🛠️ เพิ่มการ Normalize ค่าพิกเซลจาก [0, 255] เป็น [0.0, 1.0] และเปลี่ยนชนิดข้อมูลเป็น float32
    image = image.astype(np.float32) / 255.0

    return image


def to_features(images):
    # ปรับให้คืนค่าเป็น float32contiguous array สำหรับส่งเข้าโมเดล CNN
    return np.ascontiguousarray(images, dtype=np.float32)


