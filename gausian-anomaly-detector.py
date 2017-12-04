#Written By: Muhammad Rezaul Karim
# Used scikit-learn API Links:
# 1. http://scikit-learn.org/stable/modules/classes.html#

#To run experiments in python:
#1. Download and install Anaconda: https://www.anaconda.com/download/
#It Install the necessary python libraries (e.g. scikit-learn, numpy, matplotlib etc.)
#2. Open Anaconda Navigator. Launch 'jupyter notbook' to run python code


# Anomaly detection algorithm using Gaussian distribution. Use the features those follow Gaussian or normal distribution. Otherwise, transformation might require (some examples are given in this file)
# Apply it for pure anomaly detection problems only. 


# Training data sets should be named like this: train-dataset0, train-dataset1, train-dataset2 etc.
# Test data sets should be named like this: test-dataset0, test-dataset1, test-dataset2 etc.
# Make sure that file paths are correct for:  'trainingDataSetFileName' and 'testDataSetFileName' (see the pieces of code after the definition of the class: GaussianAnomalyDetector)
# configure the parameter: NUM_RUNS if you have multiple training files (e.g. train-dataset0, train-dataset1) and multiple test files

# In the current implementation, the last column in each data set must contain the decison variable, while all the other columns are used as features. 
# The value of the decison variable can be either 1 or 0. 1 means anomalous sample, while 0 means non-anomalous sample.
# If you do not need to evaluate the performance, you can remove the decison variable column for training and test files, modify the code accordingly 
# Please note this anomaly detection approach is an unsupervised learning approach. Labeling of the decision variable was done just for evaluation purpose only.

# Please make sure you configure the parameters like epsilon and NUM_RUNS


import numpy as np
import math
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import ClusterCentroids

import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
import array
import statistics

from sklearn import datasets
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


class GaussianAnomalyDetector(BaseEstimator, ClassifierMixin):

    def __init__(self, epsilon=0.05):
        self.epsilon = epsilon # epsilon is the probabilty threshold

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        self.classes_ = unique_labels(y)
        self.X_ = np.array(X)
        self.y_ = np.array(y)
  
        self.meanVector =  self.X_.mean(axis=1)  # column/feature wise based on training data. Equivalent to maximum likelihood estimation
        self.varianceVector =   self.X_.var(axis=1) # column/feature wise based on training data. Equivalent to maximum likelihood estimation
        
        return self

    def isAnomalous(self, X):

        PI=3.1416 # constant
        X=np.array(X)
        numOfSamples,numOfFeatures=X.shape
        # store the prediction for each test sample
        outputLabels=np.zeros(numOfSamples)
        
        # This part can be parallelized with threads. Not done here though. In future, I will modify it
        for i in range(numOfSamples):
            probSampleI = 1.0  # For each test sample, multiply it with the probability of each feature
            featureVector = X[i-1]

            for j in range(len(featureVector)):
              
                if self.varianceVector[j]==0:
                    self.varianceVector[j]=0.00000001 # to avoid divide by zero error
                    
                standardDeviation = math.sqrt(self.varianceVector[j])

                #compute probability of each feature with the formula of normal distribution
                squaredDiff = (featureVector[j]-self.meanVector[j])*(featureVector[j]-self.meanVector[j])
                exponentPart = math.exp(-1*(squaredDiff/(2*self.varianceVector[j])))

                otherPart = math.sqrt(2*PI)*standardDeviation
                
                if otherPart==0:
                    otherPart=0.00000001 # to avoid divide by zero error

                probFeatureJ = exponentPart/otherPart
                probSampleI = probSampleI*probFeatureJ
              

           
            if probSampleI < self.epsilon:
                outputLabels[i]=1  # 1 means anomalous
            else:
                outputLabels[i]=0  # 0 means non-anomalous
                
        #print(outputLabels)
        return  outputLabels
    
    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)
  
        return self.isAnomalous(X)
    
    def get_params(self, deep=False):
        # This estimator has parameters called "epsilon" 
        return {"epsilon": self.epsilon}

    
    
# Training data sets should be named like this: train-dataset0, train-dataset1, train-dataset2 etc.
# Test data sets should be named like this: test-dataset0, test-dataset1, test-dataset2 etc.
# Make sure that file paths are correct for:  'trainingDataSetFileName' and 'testDataSetFileName' (see below)

NUM_RUNS = 1 # The number of trials considering data at different point in time. 

    
fScoreList=[]
precScoreList=[]
recallScoreList=[]
class_names=[]
# Loop for each trial
for i in range(NUM_RUNS):
    print()
    print()
    print("###### Run " + str(i) + " has started #####")
    seed=NUM_RUNS  

    ########################### Load Data Set From File################################################################
    # Fill with anomalous data only

    trainingDataSetFileName="C:/Users/r_kar/OneDrive/Pictures/Documents/train-dataset"+ str(i) + ".csv"
    dataset1=pd.read_csv(trainingDataSetFileName)
    numOfrows1, numOfColumns1=dataset1.shape
    print("Dimension of the training data set:", dataset1.shape)  # number of rows and columns in the data set

    #Load test data (e.g. daily, weekly test data) with python panda read_csv method
   # testDataSetFileName="C:/Users/r_kar/OneDrive/Pictures/Documents/test"+ str(i) + "-anomaly.csv"
    testDataSetFileName="C:/Users/r_kar/OneDrive/Pictures/Documents/test-dataset"+ str(i) + ".csv"
    dataset2=pd.read_csv(testDataSetFileName)
    numOfrows2, numOfColumns2=dataset2.shape
    print("Dimension of the test data set:", dataset2.shape)  # number of rows and columns in the data set

    
    dataset_data_training=dataset1.iloc[ : ,0:numOfColumns1-1] #all predictor variable
    dataset_target_training=dataset1.iloc[ : ,numOfColumns1-1] # dependent variable. Assumption is that the last column contains 


    dataset_data_test=dataset2.iloc[ : ,0:numOfColumns2-1] #all predictor variable
    dataset_target_test=dataset2.iloc[ : ,numOfColumns2-1] # dependent variable. Assumption is that the last column contains 

    print()
    print("Count of samples per class (training set):")  
    print(sorted(Counter(dataset_target_training).items())) #count of different classes

    X_dataset_training,Y_dataset_training = dataset_data_training,dataset_target_training
    X_data_training = np.array(X_dataset_training)
    y_data_training = Y_dataset_training
   

   #Perform log transformation/square root transformatio/log(1+p) or any other necessary transformation of those features which do not follow normal distribution. The following pieces of code will transform all features
   #constantToBeAdded=0.0001
   #rows1, columns1= X_data_training.shape  
   #constantsArrayTraining=np.full(( rows1, columns1),constantToBeAdded)
   #X_data_training= np.log(np.add(X_data_training,constantsArrayTraining))
   #X_data_training=X_data_training**0.5  # square root transformation
   # X_data_training=np.log1p(X_data_training) # transformation with formula log(1+p)
    
    
    print("Note the number of predictors/features (second value)") 
    print(X_data_training.shape)  #reduced data set number of rows and columns

    
    #Just copy the test data here. 
    X_data_test, y_data_test = dataset_data_test,dataset_target_test
    X_data_test=np.array( X_data_test)
    y_data_test=y_data_test 
    
    
	#f you want to use subset of features instead of all features. Please make sure used indexes are relevant for your data set
    #tempIndexArray=np.array([1,2])
    #X_data_training =np.array( X_data_training)
    #X_data_training=     X_data_training[:, tempIndexArray]
	#X_data_test=X_data_test[:, tempIndexArray]

    #print(y_data_test)
        
	#Perform log transformation/square root transformation/log(1+p) or any other necessary transformation of those features which do not follow normal distribution. The following pieces of code will transform all features
	#rows1, columns1= X_data_test.shape  
    #constantsArrayTest=np.full(( rows1, columns1),constantToBeAdded)
    #X_data_test=np.log(np.add(X_data_test, constantsArrayTest))
    
 
    # Use code like this to see whether each feature follows normal distribution or not. Example shown for one column only
    plt.hist( X_data_training[:,0])
    plt.title("Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


    
    #################Fit model and predict probability #################
    classifier = GaussianAnomalyDetector()

    classifier.fit(X_data_training, y_data_training)
  

    print()
    print("predicting with the anomaly detection model:")

       
    #Predict on test data. y_pred_test contains predictions for each sample
    y_pred_test = classifier.predict(X_data_test)

   
    #print the prediction on test sets (if needed)
    print( y_pred_test)

    #get class wise precision score and store the results in a list
    #Set 'average=None' to get class wise results

    precisionResult=precision_score(y_data_test,y_pred_test,average=None)
    # Scikit learn will set the precision value to zero when no samples present for a class. 
    precScoreList.append(precisionResult)

    recallResult=recall_score(y_data_test,y_pred_test,average=None)
    # Scikit learn will set the recall value to zero when no samples present for a class. 
    recallScoreList.append(recallResult)
  
    
    #get class wise f-measure score and store the results in a list
    fScoreResult=f1_score(y_data_test,y_pred_test, average=None)
    # Scikit learn will set the F-measure value to zero when no samples present for a class. 
    fScoreList.append(fScoreResult)

        
    print()
    print()
    print("***Scikit learn will set a metric (e.g. recall) value to zero and display a warning message ") 
    print("when no samples present for a particular class in the test set***")

# For loop ends here
# Print the results of the lists that contain the results  
# print(precScoreList)
# print(recallScoreList)
# print(fScoreList)


NUM_OF_CLASSES=len(precScoreList[0])  # automatically determine the number of classes 
print(class_names)
print()  
print()  

typeOfSamples=["non-anomalous","anomalous"]
for i in range(0,NUM_OF_CLASSES):
    print()
    print("Results for the sample type:: "+ typeOfSamples[i])
    print("####################################################")
    
    # store class wise precision, recall and F-Measure
    fScoreArray=array.array("d") #type of the array
    precScoreArray=array.array("d") 
    recallScoreArray=array.array("d") 
    
    #for the current class, extract precision, recall and F-Measure values for each run/trial 
    for j in range(0,len(precScoreList)):
        # Scikit learn will set the precision value to zero when no samples present for a class. 
        precScoreArray.append(precScoreList[j][i])
        # Scikit learn will set the recall value to zero when no samples present for a class. 
        recallScoreArray.append(recallScoreList[j][i])
         # Scikit learn will set the F-measure value to zero when no samples present for a class. 
        fScoreArray.append(fScoreList[j][i])
   
 #	Enable this pieces of code if you want to verify content
 #  print(fScoreArray)
 #  print(precScoreArray)
 #  print(recallScoreArray)
    
    if(len(fScoreArray)==1): # for single trial
        print("F1 Score (Average):", fScoreArray[0])
        print("Precision Score (Average):", precScoreArray[0])
        print("Recall Score (Average):", recallScoreArray[0])
    else: # for multiple trials
        print("F1 Score (Average, Variance):", (statistics.mean(fScoreArray), statistics.variance(fScoreArray)))
        print("Precision Score (Average, Variance):",(statistics.mean(precScoreArray), statistics.variance(precScoreArray)))
        print("Recall Score (Average, Variance):", (statistics.mean(recallScoreArray), statistics.variance(recallScoreArray)))
