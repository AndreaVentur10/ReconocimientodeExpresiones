"""
    models.py - fichero dedicado a la creación de todos los modelos con los que trabajaremos. 4
    modelos pertenecen a cada una de las bases de datos (ck/fer2013/kdef/jaffe), otros 3 son
    hibridos hechos juntando de dos en dos las bases de datos ck, fer2013 y kdef (hibrido1/hibrido2/
    hibrido3), otro hibrido corresponde a la union de las tres bases de datos ck, fer2013 y kdef
    (hibrido4) y por último el modelo final (final).

    Para crear un modelo sólo basta con elegirlo al ejecutar este fichero, de la siguiente manera:
        python models.py --db final
"""
import os
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------------------------- INICIO: parser de argumentos --db database---------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--db",help="ck/fer2013/kdef/jaffe/hibrido1/hibrido2/hibrido3/hibrido4/final")
db = ap.parse_args().db

num_emotions = 7
if db == 'fer2013':
    train_dir = 'fer2013/train'
    val_dir = 'fer2013/test'
    num_train = 2124
    num_val = 570
    batch_size = 64
    num_epoch = 50
    name_m = 'models/fer2013-pre.h5'

elif db == 'jaffe':
    train_dir = 'jaffe/train'
    val_dir = 'jaffe/test'
    num_train = 170
    num_val = 43
    batch_size = 64
    num_epoch = 50
    name_m = 'models/modeljaffe.h5'

elif db == 'ck':
    train_dir = 'ck/train'
    val_dir = 'ck/test'
    num_train = 740
    num_val = 183
    batch_size = 64
    num_epoch = 50
    num_emotions = 6
    name_m = 'models/modelck.h5'

elif db == 'kdef':
    train_dir = 'kdef/train'
    val_dir = 'kdef/test'
    num_train = 784
    num_val = 196
    batch_size = 64
    num_epoch = 50
    name_m = 'models/modelkdef.h5'

elif db == 'hibrido1':
    train_dir = 'hibrido1/train'
    val_dir = 'hibrido1/test'
    num_train = 1524
    num_val = 379
    batch_size = 64
    num_epoch = 50
    name_m = 'models/modelhibrid1.h5'

elif db == 'hibrido2':
    train_dir = 'hibrido2/train'
    val_dir = 'hibrido2/test'
    num_train = 2864
    num_val = 753
    batch_size = 64
    num_epoch = 50
    name_m = 'models/modelhibrid2.h5'

elif db == 'hibrido3':
    train_dir = 'hibrido3/train'
    val_dir = 'hibrido3/test'
    num_train = 2908
    num_val = 766
    batch_size = 64
    num_epoch = 50
    name_m = 'models/modelhibrid3.h5'

elif db == 'hibrido4':
    train_dir = 'hibrido4/train'
    val_dir = 'hibrido4/test'
    num_train = 3648
    num_val = 949
    batch_size = 64
    num_epoch = 50
    name_m = 'models/modelhibrid4.h5'

elif db == 'final':
    train_dir = 'final/train'
    val_dir = 'final/test'
    num_train = 2804 # 3143
    num_val = 717 # 811
    batch_size = 64
    num_epoch = 50
    num_emotions = 6
    name_m = 'models/modelfinal2.h5'

"""   # Fer2013 sin preprocesar
    if db == 'fer2013':
        train_dir = 'fer2013/train'
        val_dir = 'fer2013/test'
        num_train = 28709
        num_val = 7178
        batch_size = 64
        num_epoch = 50
        name_m = 'models/fer2013.h5'
"""
# ------------------------- FIN: de parser de argumentos --db database---------------------------

# generadores de imagenes de entrenamiento y validación, se reescala el valor de los pixeles de
# [0,255] a [0,1]
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# agregamos el path del directorio donde se encuentran las imágenes de training. Las dividimos
# en bloques de 64 para procesar más fácilmente la cantidad de imágenes
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

# ------------------------ INICIO: Creación de Red Neuronal Convolucional --------------------------

model = Sequential()

# capa 1 convolucional 32 filtros
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
# capa 2 convolucional 64 filtros
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# capa 1 MaxPooling
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# capa 3 convolucional 128 filtros
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# capa 2 MaxPooling
model.add(MaxPooling2D(pool_size=(2, 2)))
# capa 4 convolucional 128 filtros
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# capa 3 MaxPooling
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# aplanamos el volumen resultante
model.add(Flatten())
# red neuronal con 1024 neuronas
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
# salida de X categorías: CAMBIAR A 7 o 6 !!!!!! IMPORTANTE depende de las expresiones que
# maneje el modelo
model.add(Dense(num_emotions, activation='softmax'))

# se entrena el modelo configurado con el número total de epocas (bloques) y generador de
# imágenes de validacion/testing que se utilizará para evaluar el modelo al acabar cada epoca
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])
model_info = model.fit_generator(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size)

# guardamos el modelo entrenado
model.save_weights(name_m)
