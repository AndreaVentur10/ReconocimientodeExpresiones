import argparse
import os
import glob
import cv2

# ------------------------------------- INICIO: parser de argumentos --h hibrido -------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--h",help="1/2/3/4") # 1:ck/ 2:fer2013/ 3:fer2013(pre)/ 4:jaffe
h = ap.parse_args().h

dbs = []
cnt1 = 0
cnt2 = 0

if h == "1":
    dbs.append("ck")
    dbs.append("kdef")
    hibrid = "hibrido1"
elif h == "2":
    dbs.append("ck")
    dbs.append("fer2013")
    hibrid = "hibrido2"
elif h == "3":
    dbs.append("kdef")
    dbs.append("fer2013")
    hibrid = "hibrido3"
elif h == "4":
    dbs.append("ck")
    dbs.append("kdef")
    dbs.append("fer2013")
    hibrid = "hibrido4"

# print(dbs)
# ------------------------------------- FIN: parser de argumentos --h hibrido -------------------------------------

# recorremos el array de bases de datos que vamos a hibridar
for db in dbs:
    print(db)
    # cogemos el set de entrenamiento de la base de datos
    dirs_train = glob.glob(db+"//train//*")
    # recorremos los directorios de las expresiones
    for dir in dirs_train:
        images = glob.glob(dir + "//*")
        emotion = dir.split("\\")[1]
        # se recorren todas las imágenes de cada expresión, las leemos y las guardamos en el nuevo directorio para el
        # futuro hibrido
        for image in images:
            im =cv2.imread(image)
            # print(os.path.join(hibrid + "//train", emotion, 'imh4_%s.png' % str(cnt1)))
            cv2.imwrite(os.path.join(hibrid + "//train", emotion, 'imh4_%s.png' % str(cnt1)), im)
            cnt1 += 1

    # cogemos el set de validación de la base de datos
    dirs_test = glob.glob(db + "//test//*")
    # recorremos los directorios de las expresiones
    for dir in dirs_test:
        images = glob.glob(dir + "//*")
        emotion = dir.split("\\")[1]
        # se recorren todas las imágenes de cada expresión, las leemos y las guardamos en el nuevo directorio para el
        # futuro hibrido
        for image in images:
            im =cv2.imread(image)
            cv2.imwrite(os.path.join(hibrid + "//test", emotion, 'imh4_%s.png' % str(cnt2)), im)
            cnt2 += 1
