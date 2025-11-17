
import argparse
import pathlib
import numpy as np
import tensorflow as tf
import cv2

BASE_DIR = pathlib.Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

def load_and_preprocess_image(image_path, img_size):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def main():
    parser = argparse.ArgumentParser(description="Inferencia sobre una imagen de fruta")
    parser.add_argument("--image_path", type=str, required=True, help="Ruta a la imagen a clasificar")
    parser.add_argument("--model_type", type=str, default="cnn", choices=["cnn", "mobilenet"], help="Tipo de modelo a usar")
    args = parser.parse_args()

    image_path = pathlib.Path(args.image_path)

    if args.model_type == "cnn":
        model_path = MODELS_DIR / "cnn_scratch_fruits.h5"
        img_size = (128, 128)
    else:
        model_path = MODELS_DIR / "mobilenet_fruit_quality.h5"
        img_size = (224, 224)

    print(f"Cargando modelo desde: {model_path}")
    model = tf.keras.models.load_model(model_path)

    x = load_and_preprocess_image(image_path, img_size)
    preds = model.predict(x)

    if args.model_type == "cnn":
        prob = float(preds[0][0])
        label = "madura" if prob > 0.5 else "inmadura"
        print(f"Predicción: {label} (prob={prob:.3f})")
    else:
        # clasificación multiclase
        class_idx = int(np.argmax(preds[0]))
        prob = float(np.max(preds[0]))
        class_indices = {0: "good", 1: "defective"}  # ajustar según las carpetas reales
        label = class_indices.get(class_idx, str(class_idx))
        print(f"Predicción: {label} (prob={prob:.3f})")

if __name__ == "__main__":
    main()
