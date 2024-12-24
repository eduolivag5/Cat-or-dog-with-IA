import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input

# Ruta al dataset
RUTA_DATASET = "./PetImagesFiltradas"
TAMAÑO_IMG = 100

# Listas para almacenar los datos
datos_entrenamiento = []

# Procesar las imágenes
for etiqueta, clase in enumerate(["Cat", "Dog"]):  # 0 para Cat, 1 para Dog
    ruta_clase = os.path.join(RUTA_DATASET, clase)
    for img_nombre in os.listdir(ruta_clase):
        try:
            # Ruta completa de la imagen
            img_ruta = os.path.join(ruta_clase, img_nombre)
            # Leer la imagen
            imagen = cv2.imread(img_ruta, cv2.IMREAD_COLOR)
            if imagen is None:
                continue
            # Redimensionar y procesar
            imagen = cv2.resize(imagen, (TAMAÑO_IMG, TAMAÑO_IMG))
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen = imagen.reshape(TAMAÑO_IMG, TAMAÑO_IMG, 1)
            # Añadir a los datos de entrenamiento
            datos_entrenamiento.append([imagen, etiqueta])
        except Exception as e:
            print(f"Error procesando {img_ruta}: {e}")


#Preparar mis variables X (entradas) y y (etiquetas) separadas
X = [] #imagenes de entrada (pixeles)
y = [] #etiquetas (perro o gato)

for imagen, etiqueta in datos_entrenamiento:
  X.append(imagen)
  y.append(etiqueta)

# Normalizar datos
X = np.array(X).astype(float) / 255
y = np.array(y)

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=[0.7, 1.4],
    horizontal_flip=True,
    vertical_flip=True
)

datagen.fit(X)

modeloCNN_AD = tf.keras.models.Sequential([
    Input(shape=(100, 100, 1)),  # Aquí se especifica la forma de la entrada
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

modeloCNN_AD.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])


# SEPARAR DATOS DE ENTRENAMIENTO Y DATOS DE PRUEBAS
len(X) * .85 #19700
len(X) - 19700 #3562

X_entrenamiento = X[:19700]
X_validacion = X[19700:]

y_entrenamiento = y[:19700]
y_validacion = y[19700:]

data_gen_entrenamiento = datagen.flow(X_entrenamiento, y_entrenamiento, batch_size=32)

historial = modeloCNN_AD.fit(
    data_gen_entrenamiento,
    epochs=150, batch_size=32,
    validation_data=(X_validacion, y_validacion),
    steps_per_epoch=int(np.ceil(len(X_entrenamiento) / float(32))),
    validation_steps=int(np.ceil(len(X_validacion) / float(32)))
)

print("Historial de entrenamiento:", historial.history)

# Graficar la pérdida durante el entrenamiento y validación
plt.plot(historial.history['loss'], label='Pérdida de entrenamiento')
plt.plot(historial.history['val_loss'], label='Pérdida de validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Graficar la precisión durante el entrenamiento y validación
plt.plot(historial.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(historial.history['val_accuracy'], label='Precisión de validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.show()