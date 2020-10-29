import cv2
import glob
import random
import numpy as np

emotions = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list CK DB
#emotions = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"] #Emotion list JAFFE DB
#fishface = cv2.face.createFisherFaceRecognizer() #Initialize fisher face classifier
fishface = cv2.face.FisherFaceRecognizer_create()
data = {}

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("output_ck//%s//*" % emotion) #Emotion list CK DB
    #files = glob.glob("output_jaffe//%s//*" % emotion) #Emotion list JAFFE DB
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            training_data.append(gray) #append image array to training data list
            training_labels.append(emotions.index(emotion))

        for item in prediction: #repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels

def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()

    hit = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    fail = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

    print("training fisher face classifier")
    #print("size of training set is: %d images" % len(training_labels))
    fishface.train(training_data, np.asarray(training_labels))

    print("predicting classification set")
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1
            hit[pred] = cnt
        else:
            incorrect += 1
            cnt += 1
            fail[pred] = cnt

    return ((100*correct)/(correct + incorrect)), hit, fail

#Now run it
metascore = []
percentages = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}

for i in range(0,10):
    correct, number_hits, number_fails = run_recognizer()
    #print("got %d percent correct!" % correct)
    for j in range(0,7):
        percentages[j].append( (number_hits[j] * 100)/(number_hits[j] + number_fails[j]) )
    metascore.append(correct)


print(np.mean(metascore))
print("anger: %d" % np.mean(percentages[0]))
#Emotion list CK DB
print("contempt: %d" % np.mean(percentages[1]))
print("disgust: %d" % np.mean(percentages[2]))
print("fear: %d" % np.mean(percentages[3]))
print("happy: %d" % np.mean(percentages[4]))

"""#Emotion list JAFFE DB
print("disgust: %d" % np.mean(percentages[1]))
print("fear: %d" % np.mean(percentages[2]))
print("happy: %d" % np.mean(percentages[3]))
print("neutral: %d" % np.mean(percentages[4]))"""

print("sadness: %d" % np.mean(percentages[5]))
print("surprise: %d" % np.mean(percentages[6]))

print("\n\nend score: %d percent correct!" % int(np.mean(metascore)))