
import glob
import cv2
import numpy as np

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    return faces

filename = "jaffedbase"
print("antes del bucle")
imagenes = glob.glob(filename+"/*")
#print(imagenes)
cont=0
img_pixels=48
for img in imagenes:
    print("bucle")
    print(img)
    print(img[14:16])
    emotion=""
    if img[14:16] == "AN":
        emotion = "anger"
    elif img[14:16] == "DI":
        emotion = "disgust"
    elif img[14:16] == "FE":
        emotion = "fear"
    elif img[14:16] == "HA":
        emotion = "happy"
    elif img[14:16] == "NE":
        emotion = "neutral"
    elif img[14:16] == "SA":
        emotion = "sad"
    else:
        emotion = "surprise"
    var_img = cv2.imread(img)
    face = detect_face(var_img)
    #print(face)
    if (len(face) == 0):
        continue
    for(ex, ey, ew, eh) in face:
        #de la imagen nos quedamos con la zona donde aparece el rostro
        crop_image = var_img[ey:ey+eh, ex:ex+ew]
        dim = (img_pixels, img_pixels)
        #se redimensiona a 48x48 pixels
        resize = cv2.resize(crop_image, dim)
        pixels = np.array(resize)
        #stretching del contraste de la imágen
        if pixels.max() != 255 | pixels.min() != 0:
            # print(pixels.max())
            # print(pixels.min())
            stretch = ((pixels - pixels.min())) / (pixels.max() - pixels.min())
            # print(stretch)
            stretch = stretch * 255
            # print(stretch)
        else:
            stretch = pixels
        print("entra")
        cv2.imwrite(('C:\\Users\\anvel\\Desktop\\DocumentosTFG\\Emotion-Recognition-ck+\\Emotion-Recognition-master\\output_jaffe\\%s\\output%s.jpg' % (emotion, cont)), stretch)
        cont= cont+1