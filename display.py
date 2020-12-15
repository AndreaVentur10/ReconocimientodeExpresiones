import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------------------------------------- INICIO: Red Neuronal Convolucional ----------------------------------------
model = Sequential()

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
model.add(Dense(6, activation='softmax'))

# Cargamos el modelo final
model.load_weights('models/modelfinal2.h5')  # modelo

# ---------------------------------------- FIN: Red Neuronal Convolucional ----------------------------------------

# deshabilita el uso de OpenCL
cv2.ocl.setUseOpenCL(False)

# diccionario de expresiones que maneja nuestro reconocedor de expresiones
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Happy", 3: "Neutral", 4: "Sad", 5: "Surprised"}
# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fear", 3:"Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# activamos la captura de fotogramas de la cámara
cap = cv2.VideoCapture(0)

while True:
    # Leemos fotogramas
    ret, frame = cap.read()
    if not ret:
        break
    # comenzamos cargando un clasificador en cascada para la detección del rostro
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # modificamos la escala de color RGB a GRAY para poder trabajar con la imágen
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # aplicando el clasificador facial nos quedamos con el area del rostro de la imágen
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # encuadramos el rostro a analizar
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        # hacemos la predicción de la expresión con el modelo entrenado
        prediction = model.predict(cropped_img)
        #print("prediccion:\n")
        #print(prediction)
        # nos quedamos con la probabilidad más alta del vector de 6 expresiones que devuelve la predicción
        maxindex = int(np.argmax(prediction))
        # escribimos el resultado en texto
        cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    # devolvemos la imágen capturada con la predicción
    cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
    # se cierra la ventana de retransmisión si pulsamos la tecla "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
