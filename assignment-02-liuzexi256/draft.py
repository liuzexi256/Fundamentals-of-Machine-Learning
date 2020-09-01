"""
File:   hw02.py
Author: 
Date:   
Desc:   Assignment 02B 
    
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from scipy.stats import multivariate_normal
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn import metrics

'''
Data Preprocessing
'''
# Extract data and parameters from the training set
Train_7D = np.loadtxt('CrabDatasetforTrain.txt')
Train_10D = np.loadtxt('10dDataSetforTrain.txt')
colm_size_Train_7D = np.size(Train_7D[1,:])
row_size_Train_7D = np.size(Train_7D[:,1])
colm_size_Train_10D = np.size(Train_10D[1,:])
row_size_Train_10D = np.size(Train_10D[:,1])

# Preprocess training data    
def Datapreprocess_PG_Train(X,row_size,colm_size):
    class0 = np.empty((1,colm_size))
    class1 = np.empty((1,colm_size))
    for i in np.arange(row_size):
        if X[i,colm_size-1] == 0:
            class0 = np.vstack((class0,X[i,:]))
        else:
            class1 = np.vstack((class1,X[i,:]))
    class0 = np.delete(class0,0,axis = 0)
    label0 = class0[:,colm_size-1]
    class0 = np.delete(class0,colm_size-1,axis = 1)
    
    class1 = np.delete(class1,0,axis = 0) 
    label1 = class1[:,colm_size-1]
    class1 = np.delete(class1,colm_size-1,axis = 1) 
    return class0,class1,label0,label1

def Datapreprocess_KNN_Train(X,colm_size):
    Training_Data = np.delete(X,colm_size-1,axis = 1)
    Label = X[:,colm_size-1]
    return Training_Data,Label

# Extract data from testing set
Test_7D = np.loadtxt('CrabDatasetforTest.txt')
Test_10D = np.loadtxt('10dDataSetforTest.txt')
colm_size_Test_7D = np.size(Test_7D[1,:])
row_size_Test_7D = np.size(Test_7D[:,1])
colm_size_Test_10D = np.size(Test_10D[1,:])
row_size_Test_10D = np.size(Test_10D[:,1])

# Preprocessing testing data
def Datapreprocess_Test(X,colm_size):
    test_data = np.delete(X,colm_size-1,axis = 1)
    test_label = X[:,colm_size-1]
    return test_data, test_label


'''
Train the Probabilistic Generative Model
'''
# Generate data and label for PG model
class7D_0, class7D_1, label7D_0, label7D_1 = Datapreprocess_PG_Train(Train_7D, row_size_Train_7D, colm_size_Train_7D)
class10D_0, class10D_1, label10D_0, label10D_1 = Datapreprocess_PG_Train(Train_10D, row_size_Train_10D, colm_size_Train_10D)
# Estimate the mean and covarience of training data of 7D
mu7D_0 = np.mean(class7D_0,axis=0)
cov7D_0 = np.cov(class7D_0.T)

mu7D_1 = np.mean(class7D_1, axis=0)
cov7D_1 = np.cov(class7D_1.T)


# Estimate the prior for the training data
pC7D_0 = class7D_0.shape[0]/(class7D_1.shape[0] + class7D_0.shape[0])
pC7D_1 = class7D_1.shape[0]/(class7D_1.shape[0] + class7D_0.shape[0])

# Estimate the mean and covarience of training data of 10D
mu10D_0 = np.mean(class10D_0,axis=0)
cov10D_0 = np.cov(class10D_0.T)

mu10D_1 = np.mean(class10D_1, axis=0)
cov10D_1 = np.cov(class10D_1.T)

# Estimate the prior for the training data
pC10D_0 = class10D_0.shape[0]/(class10D_1.shape[0] + class10D_0.shape[0])
pC10D_1 = class10D_1.shape[0]/(class10D_1.shape[0] + class10D_0.shape[0])


 
'''
Train the KNN Model
'''
# Generate data and label for KNN model
Training_Data_7D, Training_Label_7D = Datapreprocess_KNN_Train(Train_7D, colm_size_Train_7D)
Training_Data_10D, Training_Label_10D = Datapreprocess_KNN_Train(Train_10D, colm_size_Train_10D)
n_neighbors = 3
knn = KNeighborsClassifier(n_neighbors,weights = 'distance')

# Cross Validation for 7D
score7D = 0
for k in range(10):
    X_train7D, X_test7D, y_train7D, y_test7D = train_test_split(Training_Data_7D, Training_Label_7D, test_size = .8)
    knn.fit(X_train7D, y_train7D)
    score7D_new = knn.score(X_test7D, y_test7D)
    if score7D_new > score7D:
        score7D = score7D_new
        X_train7D_FN = X_train7D
        y_train7D_FN = y_train7D
knn_7D = knn.fit(X_train7D_FN, y_train7D_FN)
        

# Cross Validation for 10D
score10D = 0
for l in range(10):
    X_train10D, X_test10D, y_train10D, y_test10D = train_test_split(Training_Data_10D, Training_Label_10D, test_size = .8)
    knn.fit(X_train10D, y_train10D)
    score10D_new = knn.score(X_test10D, y_test10D)
    if score10D_new > score10D:
        score10D = score10D_new
        X_train10D_FN = X_train10D
        y_train10D_FN = y_train10D
knn_10D = knn.fit(X_train10D_FN, y_train10D_FN)




'''
Test the Probabilistic Genrative model
'''
# Test 7D data
test_data_7D, test_label_7D = Datapreprocess_Test(Test_7D,colm_size_Test_7D)

y7D_0 = multivariate_normal.pdf(test_data_7D, mean=mu7D_0, cov=cov7D_0, allow_singular = 'true')
y7D_1 = multivariate_normal.pdf(test_data_7D, mean=mu7D_1, cov=cov7D_1, allow_singular = 'true')

pos7D_1 = (y7D_1*pC7D_1)/(y7D_1*pC7D_1 + y7D_0*pC7D_0)
pos7D_0 = (y7D_0*pC7D_0)/(y7D_1*pC7D_1 + y7D_0*pC7D_0)

Test_result_7D = np.zeros((row_size_Test_7D,1))

for i in range(row_size_Test_7D):
    if pos7D_1[i] > pos7D_0[i]:
        Test_result_7D[i] = 1
    else:
        Test_result_7D[i] = 0


# Test 10D data
test_data_10D, test_label_10D = Datapreprocess_Test(Test_10D, colm_size_Test_10D)

y10D_0 = multivariate_normal.pdf(test_data_10D, mean=mu10D_0, cov=cov10D_0)
y10D_1 = multivariate_normal.pdf(test_data_10D, mean=mu10D_1, cov=cov10D_1)

pos10D_1 = (y10D_1*pC10D_1)/(y10D_1*pC10D_1 + y10D_0*pC10D_0)
pos10D_0 = (y10D_0*pC10D_0)/(y10D_1*pC10D_1 + y10D_0*pC10D_0)

Test_result_10D = np.zeros((row_size_Test_10D,1))

for k in range(row_size_Test_10D):
    if pos10D_1[k] > pos10D_0[k]:
        Test_result_10D[k] = 1
    else:
        Test_result_10D[k] = 0

'''
Test KNN Model
'''
# Test 7D data
test_predict7D = knn_7D.predict(test_data_7D)
print(test_predict7D)

# Test 10D data
test_score10D = knn_10D.score(test_data_10D)
