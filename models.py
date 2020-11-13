import argparse
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -----------------------------parser de argumentos --db database --m model-----------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--db",help="ck/fer2013/jaffe")
db = ap.parse_args().db

if db == 'fer2013':
    train_dir = 'fer2013/train'
    val_dir = 'fer2013/test'
    num_train = 28709
    num_val = 7178
    batch_size = 64
    num_epoch = 50
    name_m = 'fer2013.h5'

elif db == 'jaffe':
    train_dir = 'jaffe/train'
    val_dir = 'jaffe/test'
    num_train = 170
    num_val = 43
    batch_size = 64
    num_epoch = 50
    name_m = 'modeljaffe.h5'

elif db == 'ck':
    train_dir = 'ck/train'
    val_dir = 'ck/test'
    num_train = 783
    num_val = 193
    batch_size = 64
    num_epoch = 50
    name_m = 'modelck.h5'
# -----------------------------FIN de parser de argumentos --db database --m model-----------------------------

# generadores de imagenes de entrenamiento y validación, se reescala el valor de los pixeles de [0,255] a [0,1]
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# agregamos el path del directorio donde se encuentran las imágenes de training. Las dividimos en bloques de 64 para procesar más
# fácilmente la cantidad de imágenes
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

"""
Create the model:
# capa 1 convolucional 32 filtros
# capa 2 convolucional 64 filtros
# capa 1 MaxPooling
# capa 3 convolucional 128 filtros
# capa 2 MaxPooling
# capa 3 convolucional 128 filtros
# capa 3 MaxPooling
# aplanamos el volumen resultante
# red neuronal con 1024 neuronas
# salida de 7 categorías
"""
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# se entrena el modelo configurado con el número total de epocas (bloques), generador de imágenes de validacion/testing
# que se utilizará para evaluar el modelo al acabar cada epoca
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
model_info = model.fit_generator(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size)
# guardamos el modelo entrenado
model.save_weights(name_m)
