""" En este fichero python se encuentra el código fuente dedicado a los testings de cada Base de Datos(ck,jaffe,fer2013)
con cada modelo hecho a partir de dichas Bases de datos: model.h5(fer2013), modeljaffe.h5(jaffe) y modelck.h5(ck)"""

import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import cv2
import glob
import numpy as np
import os
from decimal import Decimal

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -----------------------------parser de argumentos --db database --m model-----------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--db", help="ck/fer2013/jaffe/hibrido1/hibrido2/hibrido3/hibrido4")
parser.add_argument("--m", help="ck/fer2013/jaffe/hibrido1/hibrido2/hibrido3/hibrido4")
db = parser.parse_args().db
dbmodel = parser.parse_args().m

# dependiendo de la BD que se utilice para el testing tendremos ciertas expresiones. Usaremos una lista de las
# expresiones que maneja la BD y un diccionario que traduzca el indice de la expresión del modelo de la BD por el id de
# la expresión

if db == 'ck':
    emotions = ["anger", "disgust", "fear", "happy", "sad", "surprise"]  # Emotion list CK DB
    dict_db = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 6}
else:
    emotions = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]  # GENERAL Emotion list
    dict_db = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
    # emotions = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]  # Emotion list fer2013 DB
    # dict_db = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}

# guardamos el nombre del modelo y un diccionario que traduzca el índice de la expresión del modelo de la BD por el id
# de la expresión
dict_m = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}

if dbmodel == 'ck':
    name_m = 'modelck.h5'
    dict_m = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 6}
elif dbmodel == 'jaffe':
    name_m = 'modeljaffe.h5'
elif dbmodel == 'kdef':
    name_m = 'modelkdef.h5'
elif dbmodel == 'fer2013':
    name_m = 'fer2013-pre.h5'
elif dbmodel == 'hibrido1':
    name_m = 'modelhibrid1.h5'
elif dbmodel == 'hibrido2':
    name_m = 'modelhibrid2.h5'
elif dbmodel == 'hibrido3':
    name_m = 'modelhibrid3.h5'
elif dbmodel == 'hibrido4':
    name_m = 'modelhibrid4.h5'
    # dict_m = {0: 0, 1: 1, 2: 3, 3: 4, 4: 5, 5: 6}
elif dbmodel == 'final':
    name_m = 'modelfinal2.h5'
    dict_m = {0: 0, 1: 1, 2: 3, 3: 4, 4: 5, 5: 6}

# -----------------------------FIN de parser de argumentos --db database --m model-----------------------------

dict_orig = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
# dict_orig = {0: "Angry", 1: "Disgusted", 2: "Happy", 3: "Neutral", 4: "Sad", 5: "Surprised"}

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
    hit = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    fail = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
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
        # print(pred)
        # print(pred in dict_m.keys())
        # print(dict_m[pred])
        # print(dict_db[testing_labels[cnt]])
        print("%s lo predice como : %s"% (dict_orig[dict_db[testing_labels[cnt]]],dict_orig[dict_m[pred]]))

        if dict_m[pred] == dict_db[testing_labels[cnt]]:
            correct += 1
            hit[dict_m[pred]] = hit[dict_m[pred]] + 1
            cnt += 1
        else:
            incorrect += 1
            fail[dict_db[testing_labels[cnt]]] = fail[dict_db[testing_labels[cnt]]] + 1
            cnt += 1

    return ((100 * correct) / (correct + incorrect)), hit, fail

# --------------------------------- INICIO: Cargar modelo de Red Neuronal Convolucional --------------------------------
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
model.add(Dense(7, activation='softmax')) # CAMBIAR a 6 o 7

# cargar el modelo
model.load_weights(name_m)
# ----------------------------------- FIN: Cargar modelo de Red Neuronal Convolucional ---------------------------------

#metascore = 0
percentages_exp = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

perc, number_hits, number_fails = run_recognizer()
number_exp = 0

# print(number_hits)
# print(number_fails)

# recorremos las expresiones para sacaar la precisión de acierto
for j in range(0, 7):
    try:
        res = (number_hits[j] * 100) / (number_hits[j] + number_fails[j])
        #print(res)
    except ZeroDivisionError:
        res = 0
        #print(j)
    # print(Decimal(res))
    percentages_exp[j] = res
#metascore = perc

# print(perc)
# print("porcentajes")
# print(percentages_exp)
n0 = number_hits[0] + number_fails[0]
n1 = number_hits[1] + number_fails[1]
n2 = number_hits[2] + number_fails[2]
n3 = number_hits[3] + number_fails[3]
n4 = number_hits[4] + number_fails[4]
n5 = number_hits[5] + number_fails[5]
n6 = number_hits[6] + number_fails[6]
print("Resultados Testing modelo %s con BD %s" % (dbmodel, db))
print("\nENFADO Id 0: %.2f %% | correct : %d  fail: %d" % (Decimal(np.mean(percentages_exp[0])),number_hits[0], number_fails[0] ))
print("REPUGNANCIA Id 1: %.2f %% | correct : %d  fail: %d" % (Decimal(np.mean(percentages_exp[1])), number_hits[1], number_fails[1]))
print("MIEDO Id 2: %.2f %% | correct : %d  fail: %d" % (Decimal(np.mean(percentages_exp[2])), number_hits[2], number_fails[2]))
"""print("\nfear 2: %.2f number :" % (Decimal(np.mean(percentages_exp[2]))))
print("correct : %d  fail: %d" % (number_hits[2], number_fails[2]))
print("\nhappy 3: %.2f number " % (Decimal(np.mean(percentages_exp[3]))))
print("correct : %d  fail: %d" % (number_hits[3], number_fails[3]))
print("\nneutral 4: %.2f number " % (Decimal(np.mean(percentages_exp[4]))))
print("correct : %d  fail: %d" % (number_hits[4], number_fails[4]))
print("\nsadness 5: %.2f number :" % (Decimal(np.mean(percentages_exp[5]))))
print("correct : %d  fail: %d" % (number_hits[5], number_fails[5]))
print("\nsurprise 6: %.2f number :" % (Decimal(np.mean(percentages_exp[6]))))
print("correct : %d  fail: %d" % (number_hits[6], number_fails[6]))"""
print("FELICIDAD  Id 3: %.2f %% | correct : %d  fail: %d " % (Decimal(np.mean(percentages_exp[3])), number_hits[3], number_fails[3]))
print("NEUTRAL Id 4: %.2f number %% | correct : %d  fail: %d" % (Decimal(np.mean(percentages_exp[4])), number_hits[4], number_fails[4]))
print("TRISTEZA Id 5: %.2f %% | correct : %d  fail: %d" % (Decimal(np.mean(percentages_exp[5])), number_hits[5], number_fails[5]))
print("SORPRESA Id 6: %.2f %% | correct : %d  fail: %d" % (Decimal(np.mean(percentages_exp[6])), number_hits[6], number_fails[6]))
print("\nPRECISION TOTAL DE BD: %.2f %%" % perc)

