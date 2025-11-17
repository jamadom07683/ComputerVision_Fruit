
import os
import pathlib
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------
# Configuración
# ------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "quality_train"
VAL_DIR = DATA_DIR / "quality_val"
OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# ------------------------------
# Carga de datos
# ------------------------------
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_ds = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

val_ds = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

num_classes = train_ds.num_classes

# ------------------------------
# Modelo base MobileNetV2
# ------------------------------
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)

base_model.trainable = False  # congelar para primera etapa

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ------------------------------
# Entrenamiento (fase 1)
# ------------------------------
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
)

# (Opcional) Descongelar parte del modelo base para fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_fine = model.fit(
    train_ds,
    epochs=5,
    validation_data=val_ds,
)

# ------------------------------
# Evaluación cuantitativa
# ------------------------------
val_ds.reset()
y_true = val_ds.classes
y_pred_prob = model.predict(val_ds)
y_pred = y_pred_prob.argmax(axis=1)

print("Classification report (MobileNetV2):")
print(classification_report(y_true, y_pred, target_names=list(val_ds.class_indices.keys())))
print("Matriz de confusión:")
print(confusion_matrix(y_true, y_pred))

# ------------------------------
# Guardar modelo
# ------------------------------
model.save(OUTPUT_DIR / "mobilenet_fruit_quality.h5")
print(f"Modelo guardado en: {OUTPUT_DIR / 'mobilenet_fruit_quality.h5'}")
