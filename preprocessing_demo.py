
import pathlib
import cv2
import numpy as np
import os

BASE_DIR = pathlib.Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "data" / "train"
OUTPUT_DIR = BASE_DIR / "outputs" / "preprocessing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def list_sample_images(root_dir, max_images=5):
    images = []
    for class_dir in root_dir.iterdir():
        if class_dir.is_dir():
            for img_path in class_dir.glob("*.jpg"):
                images.append(img_path)
                if len(images) >= max_images:
                    return images
    return images

def process_image(img_path, out_dir):
    img = cv2.imread(str(img_path))
    if img is None:
        return

    basename = img_path.stem

    # 1. Conversión a HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. Segmentación de color (ejemplo: tonos rojos/amarillos típicos de frutas)
    lower = np.array([0, 50, 50])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # 3. Operaciones morfológicas para limpiar la máscara
    kernel = np.ones((5, 5), np.uint8)
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)

    # 4. Detección de bordes
    edges = cv2.Canny(mask_close, 100, 200)

    # Guardar resultados
    cv2.imwrite(str(out_dir / f"{basename}_original.jpg"), img)
    cv2.imwrite(str(out_dir / f"{basename}_mask.jpg"), mask)
    cv2.imwrite(str(out_dir / f"{basename}_mask_clean.jpg"), mask_close)
    cv2.imwrite(str(out_dir / f"{basename}_edges.jpg"), edges)

def main():
    sample_images = list_sample_images(INPUT_DIR)
    if not sample_images:
        print("No se encontraron imágenes de ejemplo en", INPUT_DIR)
        return

    for img_path in sample_images:
        process_image(img_path, OUTPUT_DIR)
        print(f"Procesada {img_path.name}")

if __name__ == "__main__":
    main()
