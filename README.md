# ReconocimientodeExpresiones
TFG UPM - Reconocimiento de expresiones faciales

## Files:

ck.py - contains code for pre-processing the Cohn Kanade (CK) database

jaffe.py - contains code for pre-processing the JAFFE database

classi.py - contains code for obtaining the number of hits and fails of each expression (of a database)

testing.py - contains code for testing the models of each database with the testing images of a choosen database
              
              python testing.py --db [ck/fer2013/jaffe/hibrido1/hibrido2/hibrido3/hibrido4] --m [ck/fer2013/jaffe/hibrido1/hibrido2/hibrido3/hibrido4]
              
models.py - contains code to train the CNN models of a choosen database

              python models.py --db [fer2013/jaffe/ck/kdef]
              
hibrids.py - contains code to create the hibrid models       

               python hibrids.py --h [1/2/3/4]
                             
create_final.py - contains code to create the final database of the max values  

display.py - contains code to execute the expressions detector


haarcascade_frontalface_default.xml -  tool used for detecting faces in images


## For the execution:

Execute display.py for playing the tool with the final model trained.

                python display.py

## Database links:

-Cohn Kanade (CK) Database: https://www.kaggle.com/shawon10/ckplus

-JAFFE Database: https://zenodo.org/record/3451524#.X3o8-GgzZPY (Request access needed)

-Fer2013 Database: https://www.kaggle.com/astraszab/facial-expression-dataset-image-folders-fer2013
