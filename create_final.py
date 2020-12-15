""" En este fichero python se encuentra el código fuente dedicado a los testings de cada Base de Datos(ck,jaffe,
fer2013) con cada modelo hecho a partir de dichas Bases de datos: model.h5(fer2013), modelkdef.h5(kdef),
modeljaffe.h5(jaffe) y modelck.h5(ck) """

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import cv2
import glob
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

""" Función auxiliar make_sets() para crear 2 listas:
-testing_data: lista de todas las imágenes de todas las expresiones a testear
-testing_labels: lista de las etiquetas de estas imágenes guardadas en el mismo orden, la etiqueta será el id 
identificativo de cada expresion en el modelo"""

def make_sets(db):
    testing_data = []
    testing_labels = []
    # emotions = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]  # Emotion list DB
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

def run_recognizer(model, testing_data, testing_labels):

    #estructuras para almacenar los aciertos y fallos para cada expresión
    hit = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    fail = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    cnt = 0
    correct = 0
    incorrect = 0
    dict_orig = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

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
        # print("%s lo predice como : %s"% (dict_orig[testing_labels[cnt]],dict_orig[pred]))
        if pred == testing_labels[cnt]:
            correct += 1
            hit[pred] = hit[pred] + 1
            cnt += 1
        else:
            incorrect += 1
            fail[testing_labels[cnt]] = fail[testing_labels[cnt]] + 1
            cnt += 1

    return ((100 * correct) / (correct + incorrect)), hit, fail

def tester(name_m, testing_data, testing_labels):

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

    #metascore = 0
    percentages_exp = [0, 0, 0, 0, 0, 0, 0]

    perc, number_hits, number_fails = run_recognizer(model, testing_data, testing_labels)
    number_exp = 0
    print("got %.2f percent correct!" % perc)
    #print(number_hits)
    #print(number_fails)
    for j in range(0, 7):
        try:
            percentages_exp[j] = (number_hits[j] * 100) / (number_hits[j] + number_fails[j])
            #print(res)
        except ZeroDivisionError:
            percentages_exp[j] = 0
            #print(j)
        # print(Decimal(res))
    return np.asarray(percentages_exp)

def maxvalues():
    percentages = []
    # lista de imagenes testing y sus etiquetas, escogemos el hibrido4 ya que contiene todas las imágenes test y train
    # de todas las bases de datos
    testing_data, testing_labels = make_sets('hibrido4')
    num= 0
    for name_m in models:
        percentages.append(tester(name_m, testing_data, testing_labels))
        num += 1

    results = np.max(percentages, axis=0)
    indexs= np.argmax(percentages, axis=0)
    return results, indexs

emotions = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
models = ['models/modelhibrid1.h5', 'models/modelhibrid2.h5', 'models/modelhibrid3.h5','models/modelhibrid4.h5']
hibrids = ["hibrido1", "hibrido2", "hibrido3", "hibrido4"]
results, indexs = maxvalues()
# indexs = [1, 3, 3, 1, 2, 0, 1]
print(indexs)
# print(indexs[0])
# print(indexs[1])
# print(indexs[2])
# print(indexs[3])

# rellena el directorio "final" que contendrá los sets que mejor porcentaje de precisión han logrado para cada expresión
for n in range(0,7):
    print(n)
    print(hibrids[indexs[n]] + "//train//%s//*" % emotions[n])
    images = glob.glob(hibrids[indexs[n]] + "//train//%s//*" % emotions[n])
    cnt = 0
    for image in images:
        im = cv2.imread(image)
        # print(os.path.join(hibrid + "//train", emotion, 'imh4_%s.png' % str(cnt1)))
        cv2.imwrite(os.path.join("final//train", emotions[n], 'im_%s.png' % str(cnt)), im)
        cnt += 1

    images = glob.glob(hibrids[indexs[n]] + "//test//%s//*" % emotions[n])
    cnt = 0
    for image in images:
        im = cv2.imread(image)
        # print(os.path.join(hibrid + "//train", emotion, 'imh4_%s.png' % str(cnt1)))
        cv2.imwrite(os.path.join("final//test", emotions[n], 'im_%s.png' % str(cnt)), im)
        cnt += 1





"""
print("\nanger 0: %.2f number " % (Decimal(np.mean(percentages_exp[0]))))
print("correct : %d  fail: %d" % (number_hits[0], number_fails[0]))
print("\ndisgust 1: %.2f number :" % (Decimal(np.mean(percentages_exp[1]))))
print("correct : %d  fail: %d" % (number_hits[1], number_fails[1]))
print("\nfear 2: %.2f number :" % (Decimal(np.mean(percentages_exp[2]))))
print("correct : %d  fail: %d" % (number_hits[2], number_fails[2]))
print("\nhappy 3: %.2f number " % (Decimal(np.mean(percentages_exp[3]))))
print("correct : %d  fail: %d" % (number_hits[3], number_fails[3]))
print("\nneutral 4: %.2f number " % (Decimal(np.mean(percentages_exp[4]))))
print("correct : %d  fail: %d" % (number_hits[4], number_fails[4]))
print("\nsadness 5: %.2f number :" % (Decimal(np.mean(percentages_exp[5]))))
print("correct : %d  fail: %d" % (number_hits[5], number_fails[5]))
print("\nsurprise 6: %.2f number :" % (Decimal(np.mean(percentages_exp[6]))))
print("correct : %d  fail: %d" % (number_hits[6], number_fails[6]))
"""
