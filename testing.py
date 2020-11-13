""" En este fichero python se encuentra el código fuente dedicado a los testings de cada Base de Datos(ck,jaffe,fer2013)
con cada modelo hecho a partir de dichas Bases de datos: model.h5(fer2013), modeljaffe.h5(jaffe) y modelck.h5(ck)"""

import argparse
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import cv2
import glob
import random
import numpy as np
import os
from decimal import Decimal

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -----------------------------parser de argumentos --db database --m model-----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--db", help="ck/fer2013/jaffe")
parser.add_argument("--m", help="ck/fer2013/jaffe")
db = parser.parse_args().db
dbmodel = parser.parse_args().m

# dependiendo de la BD que se utilice para el testing tendremos ciertas expresiones. Usaremos una lista de las
# expresiones que maneja la BD y un diccionario que traduzca el indice de la expresión del modelo de la BD por el id de
# la expresión
if db == 'ck':
    emotions = ["anger", "contempt", "disgust", "fear", "happy", "sad", "surprise"]  # Emotion list CK DB
    dict_db = {0: 0, 1: 7, 2: 1, 3: 2, 4: 3, 5: 5, 6: 6}
elif db == 'jaffe':
    emotions = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]  # Emotion list JAFFE DB
    dict_db = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
else:
    emotions = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]  # Emotion list JAFFE DB
    dict_db = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}

# guardamos el nombre del modelo y un diccionario que traduzca el índice de la expresión del modelo de la BD por el id
# de la expresión
if dbmodel == 'ck':
    name_m = 'modelck.h5'
    dict_m = {0: 0, 1: 7, 2: 1, 3: 2, 4: 3, 5: 5, 6: 6}
elif dbmodel == 'jaffe':
    name_m = 'modeljaffe.h5'
    dict_m = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
else:
    name_m = 'model.h5'
    dict_m = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}


# -----------------------------FIN de parser de argumentos --db database --m model-----------------------------
# dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised", 7: "Contempt"}


""" Función auxiliar make_sets() para crear 2 listas:
-testing_data: lista de todas las imágenes de todas las expresiones a testear
-testing_labels: lista de las etiquetas de estas imágenes guardadas en el mismo orden, la etiqueta será el id 
identificativo de cada expresion en el modelo"""

def make_sets():
    testing_data = []
    testing_labels = []
    # se recorren todas las expresiones
    for emotion in emotions:
        expression_data = glob.glob(db + "//test//%s//*" % emotion)
        for item in expression_data:  # repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # cada imágen se cambia de espacio de color y se guarda en testing_data
            testing_data.append(gray)
            # se añade a la lista de etiquetas el índice de la expresión de la imágen
            testing_labels.append(emotions.index(emotion))

    return testing_data, testing_labels

""" Función auxiliar run_recognizer() para crear 2 listas y un porcentaje de aciertos general:
-hit: aciertos de cada expresión
-fail: fallos de cada expresión"""

def run_recognizer():
    #lista de imagenes testing y sus etiquetas
    testing_data, testing_labels = make_sets()
    #estructuras para almacenar los aciertos y fallos para cada expresión
    hit = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    fail = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    cnt = 0
    correct = 0
    incorrect = 0

    for image in testing_data:
        # almacenado en prediction, de ese vector
        im = np.array(image)
        # ajustamos las dimensiones de la imágen hecha array
        im_array = np.expand_dims(np.expand_dims(im, -1), 0)
        # vector de estimación
        prediction = model.predict(im_array)
        # nos quedamos con el dato más alto cuyo indice de posición corresponde al íd identificativo de la expresión
        pred = int(np.argmax(prediction))

        # si el id de la expresión es el mismo que el id de la expresión predecida se guarda el acierto sino se guarda
        # el fallo
        if dict_m[pred] == dict_db[testing_labels[cnt]]:
            correct += 1
            hit[dict_m[pred]] = hit[dict_m[pred]] + 1
            cnt += 1
        else:
            incorrect += 1
            fail[dict_db[testing_labels[cnt]]] = fail[dict_db[testing_labels[cnt]]] + 1
            cnt += 1

    return ((100 * correct) / (correct + incorrect)), hit, fail


# crear el modelo
model = Sequential()
"""
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
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
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

# cargar el modelo
model.load_weights(name_m)

metascore = []
percentages = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}

# for i in range(0,10):
correct, number_hits, number_fails = run_recognizer()
print("got %d percent correct!" % correct)
#print(number_hits)
#print(number_fails)
for j in range(0, 8):
    try:
        res = (number_hits[j] * 100) / (number_hits[j] + number_fails[j])
    except ZeroDivisionError:
        res = 0
    # print(Decimal(res))

    percentages[j].append(Decimal(res))
metascore.append(correct)

print(np.mean(metascore))
n0 = number_hits[0] + number_fails[0]
n1 = number_hits[1] + number_fails[1]
n2 = number_hits[2] + number_fails[2]
n3 = number_hits[3] + number_fails[3]
n4 = number_hits[4] + number_fails[4]
n5 = number_hits[5] + number_fails[5]
n6 = number_hits[6] + number_fails[6]
n7 = number_hits[7] + number_fails[7]
print("\nanger 0: %d number " % (Decimal(np.mean(percentages[0]))))
print("disgust 1: %d number :" % (Decimal(np.mean(percentages[1]))))
print("fear 2: %d number :" % (Decimal(np.mean(percentages[2]))))
print("happy 3: %d number " % (Decimal(np.mean(percentages[3]))))
print("neutral 4: %d number " % (Decimal(np.mean(percentages[4]))))
print("sadness 5: %d number :" % (Decimal(np.mean(percentages[5]))))
print("surprise 6: %d number :" % (Decimal(np.mean(percentages[6]))))
print("contempt 7: %d number : " % (Decimal(np.mean(percentages[7]))))

print("\n\nend score: %d percent correct!" % int(np.mean(metascore)))
