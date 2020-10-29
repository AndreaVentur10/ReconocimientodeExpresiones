
import glob
import cv2
import numpy as np

filename = "dataset"
emotions = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list "neutral"
print("antes del bucle")
for emotion in emotions:
    imagenes = glob.glob(filename+"//%s//*" % emotion)
    #print(imagenes)
    cont=0
    for img in imagenes:
        print("bucle")
        face = cv2.imread(img)
        #print(face)
        pixels = np.array(face)
        #print(pixels)
        #cv2.imshow('resized',pixels)
        #cv2.waitKey()
        #stretching del contraste de la imagen
        if pixels.max()!=255 | pixels.min()!=0:
            #print(pixels.max())
            #print(pixels.min())
            stretch = ((pixels - pixels.min())) / (pixels.max() - pixels.min())
            #print(stretch)
            stretch = stretch * 255
            #print(stretch)
        else:
            stretch = pixels

        #cv2.imshow('stretched', stretch)
        #cv2.waitKey()
        print("entra")
        cv2.imwrite(('C:\\Users\\anvel\\Desktop\\DocumentosTFG\\Emotion-Recognition-ck+\\Emotion-Recognition-master\\output_ck\\%s\\%s.jpg' % (emotion, cont)), stretch)
        cont= cont+1