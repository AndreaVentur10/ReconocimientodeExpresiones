# ReconocimientodeExpresiones
TFG UPM - Reconocimiento de expresiones faciales




https://user-images.githubusercontent.com/55163240/188217006-4eaac234-871b-4acc-8243-1408c864091a.mp4


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

## Distribution of the directories:

.

+--ck

| +--train

|    +--anger

|    +--disgust

|    +--fear

|    +--happy

|    +--neutral

|    +--sad

|    +--surprise

| +--test

|    +--anger

|    +--disgust

|    +--fear

|    +--happy

|    +--neutral

|    +--sad

|    +--surprise

+--fer2013

+--final

+--hibrido1

+--hibrido2

+--hibrido3

+--hibrido4

+--jaffe

+--kdef

+--models

| +--fer2013-pre.h5

| +--modelck.h5

| +--modelfinal2.h5

| +--modelhibrid1.h5

| +--modelhibrid2.h5

| +--modelhibrid3.h5

| +--modelhibrid4.h5

| +--modeljaffe.h5

| +--modelkdef.h5

+--ck.py

+--classi.py

+--create_final.py

+--display.py

+--haarcascade_frontalface_default.xml

+--hibrids.py

+--jaffe.py

+--models.py

+--README.md

+--testing.py
