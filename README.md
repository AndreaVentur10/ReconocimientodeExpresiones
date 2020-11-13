# ReconocimientodeExpresiones
TFG UPM - Reconocimiento de expresiones faciales

## Files:

ck.py - contains code for pre-processing the Cohn Kanade (CK) database

jaffe.py - contains code for pre-processing the JAFFE database

classi.py - contains code for obtaining the number of hits and fails of each expression (of a database)

testing.py - contains code for testing the models of each database with the testing images of a choosen database
              
              python testing.py --db [fer2013/jaffe/ck] --m [fer2013/jaffe/ck]

haarcascade_frontalface_default.xml -  tool used for detecting faces in images



## Database links:

-Cohn Kanade (CK) Database: https://www.kaggle.com/shawon10/ckplus

-JAFFE Database: https://zenodo.org/record/3451524#.X3o8-GgzZPY (Request access needed)

-Fer2013 Database: https://www.kaggle.com/astraszab/facial-expression-dataset-image-folders-fer2013
