
FruitVision — Clasificación de Frutas Frescas y Podridas con Técnicas de Visión por Computador

Autor:

Jesús Santiago Amado Montaña

Objetivo del proyecto

Desarrollar un sistema capaz de clasificar frutas frescas y podridas usando modelos de Visión por Computador, aplicando:

Entrenamiento de un modelo desde cero

Transfer learning con MobileNetV2

Preprocesamiento y limpieza de imágenes

Segmentación y análisis de color

Inferencia sobre imágenes nuevas

Dataset:

Se utiliza el dataset público:

“Fruits Fresh and Rotten for Classification”
https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

Incluye 6 clases:

freshapples

freshbanana

freshoranges

rottenapples

rottenbanana

rottenoranges

Modelos implementados
1. Modelo CNN desde cero

Archivo: train_cnn_scratch.py

Incluye:

Capas Conv2D

MaxPooling

Flatten

Dense

Regularización y augmentación

2. Transfer Learning con MobileNetV2

Archivo: train_transfer_mobilenet.py

Incluye:

Carga de MobileNetV2 sin top

Congelamiento de capas

Fine-tuning

Aumento de datos

Resultado típico:

Accuracy entre 93% y 96%

Preprocesamiento:

Archivo: preprocessing_demo.py

Técnicas incluidas:

Redimensionamiento

Normalización

Eliminación de ruido

Aumento de datos

Comparación “antes/después”

Inferencia

Archivo: infer_image.py

Permite:

Cargar el modelo entrenado

Darle la ruta de una imagen y obtener:

clase predicha

probabilidad

visualización opcional
