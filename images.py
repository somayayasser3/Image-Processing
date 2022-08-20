# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 23:03:20 2021

@author: acer
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 19:27:33 2021

@author: acer
"""

import cv2
import glob
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def svmClass(features,output,featuresO,testO):
    
    clf = svm.SVC(kernel='linear')
    clf.fit(features, output)
    outA = clf.predict(featuresO)
    acc = accuracy_score(testO, outA) * 100
    pre = precision_score(testO, outA,average='micro')
    rec = recall_score(testO, outA,average='micro')
    return pre, rec, acc




def naiveBayes(features, output,featuresO,testO):
   
    gnb = GaussianNB()
    gnb.fit(features, output)
    y_pred = gnb.predict(featuresO)
    acc = accuracy_score(testO, y_pred) * 100
    pre = precision_score(testO, y_pred,average='micro')
    rec = recall_score(testO, y_pred,average='micro')
    return pre, rec, acc



def logClass(features, output,featuresO,testO):

    classifier = LogisticRegression(random_state=0)
    classifier.fit(features, output)
    outA = classifier.predict(featuresO)
    acc = accuracy_score(testO, outA) * 100
    pre = precision_score(testO, outA,average='micro')
    rec = recall_score(testO, outA,average='micro')
    return pre, rec, acc


def flatten(imageNames):
    labelOutput = []

    readImagesGray = []
    readImagesBinary = []
    readImagesRGB = []
    x=""
    readImagesGrayCanny = []
    readImagesBinaryCanny = []
    readImagesRGBCanny = []
    images = imageNames[:100] + imageNames[80000: 80100] + imageNames[40000:40100]
    for image in images:
        imgRGB = cv2.imread(image)
        imgRGB=cv2.resize(imgRGB,(200,200))
        imgRGB = cv2.resize(imgRGB, (0, 0), fx=0.25, fy=0.25)
        readImagesRGB.append(imgRGB)
        imgBlurRGB = cv2.GaussianBlur(imgRGB, (3, 3), 0)
        edges = cv2.Canny(image=imgBlurRGB, threshold1=4, threshold2=100)  # Canny Edge Detection
        readImagesRGBCanny.append(edges)

        imgG = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        imgG=cv2.resize(imgG,(200,200))
        imgG = cv2.resize(imgG, (0, 0), fx=0.25, fy=0.28)
        
        readImagesGray.append(imgG)
        imgBlurGray = cv2.GaussianBlur(imgG, (3, 3), 0)
        edgesGray = cv2.Canny(image=imgBlurGray, threshold1=4, threshold2=100)  # Canny Edge Detection
        readImagesGrayCanny.append(edgesGray)
        x=imgG
        img = cv2.imread(image, 2)
        img=cv2.resize(img,(200,200))
        img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        r, threshold = cv2.threshold(img, 149, 255, cv2.THRESH_BINARY)
        readImagesBinary.append(threshold)
        imgBlurBinary = cv2.GaussianBlur(img, (3, 3), 0)
        edgesBinary = cv2.Canny(image=imgBlurBinary, threshold1=4, threshold2=100)  # Canny Edge Detection
        readImagesBinaryCanny.append(edgesBinary)
      

    lenofimage=len(images)
    readImagesGrayCanny=np.array(readImagesGrayCanny)
    readImagesRGBCanny=np.array(readImagesRGBCanny)
    readImagesBinaryCanny=np.array(readImagesBinaryCanny)
    print(readImagesBinaryCanny.shape)
   
    
    flattenedRGB = np.array(readImagesRGBCanny).reshape(lenofimage,-1)
    flattenedGray = np.array(readImagesGrayCanny).reshape(lenofimage,-1)
    flattenedBinary =np.array(readImagesBinaryCanny).reshape(lenofimage,-1)
    print(flattenedBinary.shape)
   
    
    return flattenedBinary, flattenedGray, flattenedRGB


directory = glob.glob(
   'H:/college/level_four/Machine Learning2/assignment2/ASL_Alphabet_Dataset/asl_alphabet_train/*')
imageNamesTrain = []
labelOutputTrain = []
for folder in directory:
    for file in glob.glob(folder + '/*.jpg'):
        # print(file)
        imageNamesTrain.append(file)

imageTrain = imageNamesTrain[:100] + imageNamesTrain[80000: 80100] + imageNamesTrain[40000:40100]
for image in imageTrain:
    labels = image.split("/")  
    x=labels[6].split("\\")
    name = x[1]
    labelOutputTrain.append(name)

labelOutputTest = []
imageNamesTest = []
for file in glob.glob('H:/college/level_four/Machine Learning2/assignment2/ASL_Alphabet_Dataset/asl_alphabet_test/*.jpg'):
    imageNamesTest.append(file)
    

for image in imageNamesTest:
    labels = image.split("/")[-1]
    #print(labels)
    name = labels.split("\\")[1]
    finLabel = name.split("_")[0]
    labelOutputTest.append(finLabel)

#print(labelOutputTrain)
trainB,trainG,trainRGB = flatten(imageNamesTrain)
testB,testG,testRGB = flatten(imageNamesTest)

listTrainCases=[trainB,trainG,trainRGB] 
listTestCases=[testB,testG,testRGB]

for i in range(3):

    print(svmClass(listTrainCases[i],labelOutputTrain,listTestCases[i],labelOutputTest))
    print("##########################################")
    print(logClass(listTrainCases[i],labelOutputTrain,listTestCases[i],labelOutputTest))
    print("############################################################3")
    print(naiveBayes(listTrainCases[i],labelOutputTrain,listTestCases[i],labelOutputTest))
    print("############################################################")