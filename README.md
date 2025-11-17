
# Proyecto: Detección automática de madurez y calidad de frutas

Este proyecto implementa un sistema de visión por computador para clasificar frutas según su estado de madurez
y presencia de defectos superficiales, usando modelos entrenados desde cero y modelos preentrenados
(con Transfer Learning).

## Estructura del proyecto

- `train_cnn_scratch.py`: Entrenamiento de una CNN desde cero para clasificar frutas (madura vs inmadura).
- `train_transfer_mobilenet.py`: Entrenamiento de un modelo basado en MobileNetV2 para clasificar frutas según calidad.
- `infer_image.py`: Script de inferencia para clasificar una imagen individual usando un modelo entrenado.
- `preprocessing_demo.py`: Ejemplos de preprocesamiento con OpenCV (HSV, segmentación por color, bordes, morfología).
- `requirements.txt`: Librerías principales utilizadas.

## Organización de datos esperada

Los scripts de entrenamiento asumen una estructura de carpetas como:

```
data/
    train/
        ripe/
        unripe/
    val/
        ripe/
        unripe/
    quality_train/
        good/
        defective/
    quality_val/
        good/
        defective/
```

Cada subcarpeta contiene imágenes JPG/PNG de la clase correspondiente.

## Cómo usar

1. Crear y activar un entorno virtual de Python.
2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Entrenar la CNN desde cero:

```bash
python train_cnn_scratch.py
```

4. Entrenar el modelo con MobileNetV2:

```bash
python train_transfer_mobilenet.py
```

5. Ejecutar inferencia sobre una imagen:

```bash
python infer_image.py --image_path ruta/a/imagen.jpg --model_type mobilenet
```

## Notas

- El código está pensado como base académica: puede ajustarse el número de épocas, tamaño del batch
  y rutas de datos según el entorno del estudiante.
- Para la sustentación, se recomienda mostrar ejemplos de imágenes de entrada, salidas de los modelos,
  matrices de confusión y curvas de entrenamiento.
