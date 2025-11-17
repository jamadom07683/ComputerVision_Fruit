
import os
import pathlib
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ------------------------------
# Configuración general
# ------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15  # se puede ajustar en la sustentación

# ------------------------------
# Carga de datos
# ------------------------------
train_ds = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
).flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

val_ds = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
).flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

# ------------------------------
# Definición del modelo CNN desde cero
# ------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=IMG_SIZE + (3,)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ------------------------------
# Entrenamiento
# ------------------------------
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
)

# ------------------------------
# Evaluación cuantitativa
# ------------------------------
val_ds.reset()
y_true = val_ds.classes
y_pred_prob = model.predict(val_ds)
y_pred = (y_pred_prob > 0.5).astype("int32").ravel()

print("Classification report (CNN desde cero):")
print(classification_report(y_true, y_pred, target_names=list(val_ds.class_indices.keys())))

print("Matriz de confusión:")
print(confusion_matrix(y_true, y_pred))

# ------------------------------
# Guardar modelo
# ------------------------------
model.save(OUTPUT_DIR / "cnn_scratch_fruits.h5")
print(f"Modelo guardado en: {OUTPUT_DIR / 'cnn_scratch_fruits.h5'}")
