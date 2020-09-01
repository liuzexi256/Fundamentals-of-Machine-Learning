"""
File:   hw02.py
Author: Zexi Liu
Date:   
Desc:   Assignment 02B 
    
"""


""" =======================  Import dependencies ========================== """
import numpy as np 
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc 


""" =======================  Function definitions  ========================== """
def cross_validationcrab_pg(traindata, classifydata):
    score = 0
    scoresum = 0
    for i in range(10):
        train_datafortrain_train, train_datafortrain_test, train_dataforcls_train_train, train_dataforcls_train_test = train_test_split(traindata, classifydata, test_size=.2)

        N1 = 0
        N0 = 0

        for i in range(len(train_dataforcls_train_train)):
            if train_dataforcls_train_train[i] == 1:
                N1 = N1 + 1
            else:
                N0 = N0 + 1
        train_dataT = np.zeros((N1,7))
        train_dataF = np.zeros((N0,7))
        N11 = 0
        N00 = 0
        for i in range(len(train_dataforcls_train_train)):
            if train_dataforcls_train_train[i] == 1:
                train_dataT[N11,0] = train_datafortrain_train[i,0]
                train_dataT[N11,1] = train_datafortrain_train[i,1]
                train_dataT[N11,2] = train_datafortrain_train[i,2]
                train_dataT[N11,3] = train_datafortrain_train[i,3]
                train_dataT[N11,4] = train_datafortrain_train[i,4]
                train_dataT[N11,5] = train_datafortrain_train[i,5]
                train_dataT[N11,6] = train_datafortrain_train[i,6]
                N11 = N11 + 1
            else:
                train_dataF[N00,0] = train_datafortrain_train[i,0]
                train_dataF[N00,1] = train_datafortrain_train[i,1]
                train_dataF[N00,2] = train_datafortrain_train[i,2]
                train_dataF[N00,3] = train_datafortrain_train[i,3]
                train_dataF[N00,4] = train_datafortrain_train[i,4]
                train_dataF[N00,5] = train_datafortrain_train[i,5]
                train_dataF[N00,6] = train_datafortrain_train[i,6]
                N00 = N00 + 1

        mu1 = np.mean(train_dataT, axis=0)
        mu0 = np.mean(train_dataF, axis=0)
        cov1 = np.cov(train_dataT.T)
        cov0 = np.cov(train_dataF.T)

        la = 0.1
        for i in range(len(cov1[0,:])):
            cov1[i,i] = cov1[i,i] + la
        for i in range(len(cov0[0,:])):
            cov0[i,i] = cov0[i,i] + la

        p1 = N1/len(train_dataforcls_train_train)
        p0 = N0/len(train_dataforcls_train_train)

        y1 = multivariate_normal.pdf(train_datafortrain_test, mean=mu1, cov=cov1, allow_singular=False)
        y0 = multivariate_normal.pdf(train_datafortrain_test, mean=mu0, cov=cov0, allow_singular=False)
        pos1 = (y1*p1)/(y1*p1 + y0*p0)
        pos0 = (y0*p0)/(y1*p1 + y0*p0)



        classify = np.zeros((len(pos1),1))
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for i in range(len(pos1)):
            if pos1[i] > pos0[i]:
                classify[i,0] = 1
                if classify[i,0] == train_dataforcls_train_test[i]:
                    tp = tp + 1
                else:
                    fp = fp + 1
            else:
                classify[i,0] = 0
                if classify[i,0] == train_dataforcls_train_test[i]:
                    tn = tn + 1
                else:
                    fn = fn + 1
        score = (tp + tn)/(tp + fp + fn + tn)
        scoresum = scoresum + score
    scoremean = scoresum/10
    return(scoremean)

def cross_validation10d_pg(traindata, classifydata):
    score = 0
    scoresum = 0
    for i in range(10):
        train_datafortrain_train, train_datafortrain_test, train_dataforcls_train_train, train_dataforcls_train_test = train_test_split(traindata, classifydata, test_size=.2)

        N1 = 0
        N0 = 0

        for i in range(len(train_dataforcls_train_train)):
            if train_dataforcls_train_train[i] == 1:
                N1 = N1 + 1
            else:
                N0 = N0 + 1
        train_dataT = np.zeros((N1,10))
        train_dataF = np.zeros((N0,10))
        N11 = 0
        N00 = 0
        for i in range(len(train_dataforcls_train_train)):
            if train_dataforcls_train_train[i] == 1:
                train_dataT[N11,0] = train_datafortrain_train[i,0]
                train_dataT[N11,1] = train_datafortrain_train[i,1]
                train_dataT[N11,2] = train_datafortrain_train[i,2]
                train_dataT[N11,3] = train_datafortrain_train[i,3]
                train_dataT[N11,4] = train_datafortrain_train[i,4]
                train_dataT[N11,5] = train_datafortrain_train[i,5]
                train_dataT[N11,6] = train_datafortrain_train[i,6]
                train_dataT[N11,7] = train_datafortrain_train[i,7]
                train_dataT[N11,8] = train_datafortrain_train[i,8]
                train_dataT[N11,9] = train_datafortrain_train[i,9]
                N11 = N11 + 1
            else:
                train_dataF[N00,0] = train_datafortrain_train[i,0]
                train_dataF[N00,1] = train_datafortrain_train[i,1]
                train_dataF[N00,2] = train_datafortrain_train[i,2]
                train_dataF[N00,3] = train_datafortrain_train[i,3]
                train_dataF[N00,4] = train_datafortrain_train[i,4]
                train_dataF[N00,5] = train_datafortrain_train[i,5]
                train_dataF[N00,6] = train_datafortrain_train[i,6]
                train_dataF[N00,7] = train_datafortrain_train[i,7]
                train_dataF[N00,8] = train_datafortrain_train[i,8]
                train_dataF[N00,9] = train_datafortrain_train[i,9]
                N00 = N00 + 1

        mu1 = np.mean(train_dataT, axis=0)
        mu0 = np.mean(train_dataF, axis=0)
        cov1 = np.cov(train_dataT.T)
        cov0 = np.cov(train_dataF.T)
        p1 = N1/len(train_dataforcls_train_train)
        p0 = N0/len(train_dataforcls_train_train)

        y1 = multivariate_normal.pdf(train_datafortrain_test, mean=mu1, cov=cov1, allow_singular=False)
        y0 = multivariate_normal.pdf(train_datafortrain_test, mean=mu0, cov=cov0, allow_singular=False)
        pos1 = (y1*p1)/(y1*p1 + y0*p0)
        pos0 = (y0*p0)/(y1*p1 + y0*p0)

        classify = np.zeros((len(pos1),1))
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for i in range(len(pos1)):
            if pos1[i] > pos0[i]:
                classify[i,0] = 1
                if classify[i,0] == train_dataforcls_train_test[i]:
                    tp = tp + 1
                else:
                    fp = fp + 1
            else:
                classify[i,0] = 0
                if classify[i,0] == train_dataforcls_train_test[i]:
                    tn = tn + 1
                else:
                    fn = fn + 1
        score = (tp + tn)/(tp + fp + fn + tn)
        scoresum = scoresum + score
    scoremean = scoresum/10
    return(scoremean)

def plot1gp(data):
    y1 = multivariate_normal.pdf(data, mean=mu1_10d, cov=cov1_10d, allow_singular=False)
    y0 = multivariate_normal.pdf(data, mean=mu0_10d, cov=cov0_10d, allow_singular=False)
    pos1 = (y1*p1_10d)/(y1*p1_10d + y0*p0_10d)
    pos0 = (y0*p0_10d)/(y1*p1_10d + y0*p0_10d)

    classify = np.zeros((len(pos1),1))
    N1 = 0
    N0 = 0

    for i in range(len(pos1)):
        if pos1[i] > pos0[i]:
            classify[i,0] = 1
            N1 = N1 + 1
        else:
            classify[i,0] = 0
            N0 = N0 + 1
    t = np.zeros((N1,10))
    f = np.zeros((N0,10))

    N11 = 0
    N00 = 0
    for i in range(len(classify)):
        if classify[i] == 1:
            t[N11,4] = train_data2[i,4]
            t[N11,5] = train_data2[i,5]
            N11 = N11 + 1
        else:
            f[N00,4] = train_data2[i,4]
            f[N00,5] = train_data2[i,5]
            N00 = N00 + 1
    return(classify)

def plot3_10d(testdata, classifydata):
    k = np.arange(0,31)
    accuracy = np.zeros(31)
    for n_neighbors in range(1,31):
        classifiers = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
        classifiers.fit(train_data2fortrain, train_data2forcls)

        Z = classifiers.predict(testdata)

        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for i in range(len(pos1_10d)):
            if Z[i] == 1:
                if Z[i] == classifydata[i]:
                    tp = tp + 1
                else:
                    fp = fp + 1
            else:
                if Z[i] == classifydata[i]:
                    tn = tn + 1
                else:
                    fn = fn + 1
        accuracy[n_neighbors] = (tp + tn)/(tp + fp + fn + tn)
    return(k,accuracy)

def plot3_10d_train(testdata, classifydata):
    k = np.arange(0,31)
    accuracy = np.zeros(31)
    for n_neighbors in range(1,31):
        score = 0
        scoresum = 0
        for i in range(0,10):
            train_datafortrain_train, train_datafortrain_test, train_dataforcls_train, train_dataforcls_test = train_test_split(testdata, classifydata, test_size=.2)
           
            classifiers = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
       	    classifiers.fit(train_datafortrain_train, train_dataforcls_train)

            Z = classifiers.predict(train_datafortrain_test)

            tp = 0
            fp = 0
            fn = 0
            tn = 0
            for i in range(len(train_dataforcls_test)):
                if Z[i] == 1:
                    if Z[i] == train_dataforcls_test[i]:
                        tp = tp + 1
                    else:
                        fp = fp + 1
                else:
                    if Z[i] == train_dataforcls_test[i]:
                        tn = tn + 1
                    else:
                        fn = fn + 1
            score = (tp + tn)/(tp + fp + fn + tn)
            scoresum = scoresum + score
        accuracy[n_neighbors] = scoresum/10
    return(k,accuracy)


""" =======================  Load Crab Training Data ======================= """
train_data1= np.loadtxt('CrabDatasetforTrain.txt')
train_data1fortrain = train_data1[:,0:7]
train_data1forcls = train_data1[:,7]

N1_crab = 0
N0_crab = 0

for i in range(len(train_data1forcls)):
    if train_data1forcls[i] == 1:
        N1_crab = N1_crab + 1
    else:
        N0_crab = N0_crab + 1
train_data1T = np.zeros((N1_crab,7))
train_data1F = np.zeros((N0_crab,7))
N11_crab = 0
N00_crab = 0
for i in range(len(train_data1forcls)):
    if train_data1forcls[i] == 1:
        train_data1T[N11_crab,0] = train_data1[i,0]
        train_data1T[N11_crab,1] = train_data1[i,1]
        train_data1T[N11_crab,2] = train_data1[i,2]
        train_data1T[N11_crab,3] = train_data1[i,3]
        train_data1T[N11_crab,4] = train_data1[i,4]
        train_data1T[N11_crab,5] = train_data1[i,5]
        train_data1T[N11_crab,6] = train_data1[i,6]
        N11_crab = N11_crab + 1
    else:
        train_data1F[N00_crab,0] = train_data1[i,0]
        train_data1F[N00_crab,1] = train_data1[i,1]
        train_data1F[N00_crab,2] = train_data1[i,2]
        train_data1F[N00_crab,3] = train_data1[i,3]
        train_data1F[N00_crab,4] = train_data1[i,4]
        train_data1F[N00_crab,5] = train_data1[i,5]
        train_data1F[N00_crab,6] = train_data1[i,6]
        N00_crab = N00_crab + 1


""" =======================  Load 10D Training Data ======================= """
train_data2= np.loadtxt('10dDatasetforTrain.txt')
train_data2fortrain = train_data2[:,0:10]
train_data2forcls = train_data2[:,10]

N1_10d = 0
N0_10d = 0

for i in range(len(train_data2forcls)):
    if train_data2forcls[i] == 1:
        N1_10d = N1_10d + 1
    else:
        N0_10d = N0_10d + 1
train_data2T = np.zeros((N1_10d,10))
train_data2F = np.zeros((N0_10d,10))
N11_10d = 0
N00_10d = 0
for i in range(len(train_data2forcls)):
    if train_data2forcls[i] == 1:
        train_data2T[N11_10d,0] = train_data2[i,0]
        train_data2T[N11_10d,1] = train_data2[i,1]
        train_data2T[N11_10d,2] = train_data2[i,2]
        train_data2T[N11_10d,3] = train_data2[i,3]
        train_data2T[N11_10d,4] = train_data2[i,4]
        train_data2T[N11_10d,5] = train_data2[i,5]
        train_data2T[N11_10d,6] = train_data2[i,6]
        train_data2T[N11_10d,7] = train_data2[i,7]
        train_data2T[N11_10d,8] = train_data2[i,8]
        train_data2T[N11_10d,9] = train_data2[i,9]
        N11_10d = N11_10d + 1
    else:
        train_data2F[N00_10d,0] = train_data2[i,0]
        train_data2F[N00_10d,1] = train_data2[i,1]
        train_data2F[N00_10d,2] = train_data2[i,2]
        train_data2F[N00_10d,3] = train_data2[i,3]
        train_data2F[N00_10d,4] = train_data2[i,4]
        train_data2F[N00_10d,5] = train_data2[i,5]
        train_data2F[N00_10d,6] = train_data2[i,6]
        train_data2F[N00_10d,7] = train_data2[i,7]
        train_data2F[N00_10d,8] = train_data2[i,8]
        train_data2F[N00_10d,9] = train_data2[i,9]
        N00_10d = N00_10d + 1
""" ========================  Train the Crab Model by Probabilistic Generative Model ============================= """
mu1_crab = np.mean(train_data1T, axis=0)
mu0_crab = np.mean(train_data1F, axis=0)
cov1_crab = np.cov(train_data1T.T)
cov0_crab = np.cov(train_data1F.T)

la = 0.1
for i in range(len(cov1_crab[0,:])):
    cov1_crab[i,i] = cov1_crab[i,i] + la
for i in range(len(cov0_crab[0,:])):
    cov0_crab[i,i] = cov0_crab[i,i] + la

p1_crab = N1_crab/len(train_data1forcls)
p0_crab = N0_crab/len(train_data1forcls)

""" ========================  Train the 10D Model by Probabilistic Generative Model ============================= """
mu1_10d = np.mean(train_data2T, axis=0)
mu0_10d = np.mean(train_data2F, axis=0)
cov1_10d = np.cov(train_data2T.T)
cov0_10d = np.cov(train_data2F.T)
p1_10d = N1_10d/len(train_data2forcls)
p0_10d = N0_10d/len(train_data2forcls)


""" ========================  Train the Crab Model by K-Nearest Neighbors Algorithm ============================= """
n_neighbors_crab = 3
classifiers_crab = []
classifiers_crab.append(neighbors.KNeighborsClassifier(n_neighbors_crab, weights='uniform'))
classifiers_crab.append(neighbors.KNeighborsClassifier(n_neighbors_crab, weights='distance'))
names_crab = ['K-NN_Uniform', 'K-NN_Weighted']
scoremax_crab = 0
scoresum_crab = 0
socremean_crab = 0
for i in range(0,10):
    train_data1fortrain_train, train_data1fortrain_test, train_data1forcls_train, train_data1forcls_test = train_test_split(train_data1fortrain, train_data1forcls, test_size=.2)

    for name, clf in zip(names_crab, classifiers_crab):
        clf.fit(train_data1fortrain_train, train_data1forcls_train)
        score_crab = clf.score(train_data1fortrain_test, train_data1forcls_test)
        if score_crab > scoremax_crab:
            scoremax_crab = score_crab
            imax_crab = i
            clffinal_crab = clf.fit(train_data1fortrain_train, train_data1forcls_train)
    scoresum_crab = scoresum_crab + scoremax_crab
socremean_crab = scoresum_crab/10


""" ========================  Train the 10D Model by K-Nearest Neighbors Algorithm ============================= """
n_neighbors_10d = 3
classifiers_10d = []
classifiers_10d.append(neighbors.KNeighborsClassifier(n_neighbors_10d, weights='uniform'))
classifiers_10d.append(neighbors.KNeighborsClassifier(n_neighbors_10d, weights='distance'))
names_10d = ['K-NN_Uniform', 'K-NN_Weighted']
scoremax_10d = 0
scoresum_10d = 0
socremean_10d = 0
for i in range(0,10):
    train_data2fortrain_train, train_data2fortrain_test, train_data2forcls_train, train_data2forcls_test = train_test_split(train_data2fortrain, train_data2forcls, test_size=.2)

    for name, clf in zip(names_10d, classifiers_10d):
        clf.fit(train_data2fortrain_train, train_data2forcls_train)
        score_10d = clf.score(train_data2fortrain_test, train_data2forcls_test)
        if score_10d > scoremax_10d:
            scoremax_10d = score_10d
            imax_10d = i
            clffinal_10d = clf.fit(train_data2fortrain_train, train_data2forcls_train)
    scoresum_10d = scoresum_10d + scoremax_10d
socremean_10d = scoresum_10d/10


""" ======================================= Cross-validation for PG ========================================== """
socre_pg_crab = cross_validationcrab_pg(train_data1fortrain, train_data1forcls)
socre_pg_10d = cross_validation10d_pg(train_data2fortrain, train_data2forcls)


""" ======================================= Load Crab Test Data ========================================== """
test_data1 = np.loadtxt('CrabDatasetforTest.txt')

test_data1fortest = test_data1[:,0:7]
test_data1forcls = test_data1[:,7]

""" ======================================= Load 10D Test Data ========================================== """
test_data2 = np.loadtxt('10dDataSetforTest.txt')
test_data2forcls = test_data2[:,10]
test_data2fortest = test_data2[:,0:10]
N1_10dtest = 0
N0_10dtest = 0
for i in range(len(test_data2forcls)):
    if test_data2forcls[i] == 1:
        N1_10dtest = N1_10dtest + 1
    else:
        N0_10dtest = N0_10dtest + 1
test_data2T = np.zeros((N1_10dtest,10))
test_data2F = np.zeros((N0_10dtest,10))
N11_10dtest = 0
N00_10dtest = 0
for i in range(len(test_data2forcls)):
    if test_data2forcls[i] == 1:
        test_data2T[N11_10dtest,4] = test_data2[i,4]
        test_data2T[N11_10dtest,5] = test_data2[i,5]
        N11_10dtest = N11_10dtest + 1
    else:
        test_data2F[N00_10dtest,4] = test_data2[i,4]
        test_data2F[N00_10dtest,5] = test_data2[i,5]
        N00_10dtest = N00_10dtest + 1



""" ============================= Test Crab Probabilistic Generative Model =============================== """
y1_crab = multivariate_normal.pdf(test_data1fortest, mean=mu1_crab, cov=cov1_crab, allow_singular=False)
y0_crab = multivariate_normal.pdf(test_data1fortest, mean=mu0_crab, cov=cov0_crab, allow_singular=False)
pos1_crab = (y1_crab*p1_crab)/(y1_crab*p1_crab + y0_crab*p0_crab)
pos0_crab = (y0_crab*p0_crab)/(y1_crab*p1_crab + y0_crab*p0_crab)



classify_crab = np.zeros((len(pos1_crab),1))
tp_crab_PG = 0
fp_crab_PG = 0
fn_crab_PG = 0
tn_crab_PG = 0
for i in range(len(pos1_crab)):
    if pos1_crab[i] > pos0_crab[i]:
        classify_crab[i,0] = 1
        if classify_crab[i,0] == test_data1[i,7]:
            tp_crab_PG = tp_crab_PG + 1
        else:
            fp_crab_PG = fp_crab_PG + 1
    else:
        classify_crab[i,0] = 0
        if classify_crab[i,0] == test_data1[i,7]:
            tn_crab_PG = tn_crab_PG + 1
        else:
            fn_crab_PG = fn_crab_PG + 1

""" ============================= Test 10D Probabilistic Generative Model =============================== """
y1_10d = multivariate_normal.pdf(test_data2fortest, mean=mu1_10d, cov=cov1_10d, allow_singular=False)
y0_10d = multivariate_normal.pdf(test_data2fortest, mean=mu0_10d, cov=cov0_10d, allow_singular=False)
pos1_10d = (y1_10d*p1_10d)/(y1_10d*p1_10d + y0_10d*p0_10d)
pos0_10d = (y0_10d*p0_10d)/(y1_10d*p1_10d + y0_10d*p0_10d)



classify_10d = np.zeros((len(pos1_10d),1))
tp_10d_PG = 0
fp_10d_PG = 0
fn_10d_PG = 0
tn_10d_PG = 0
for i in range(len(pos1_10d)):
    if pos1_10d[i] > pos0_10d[i]:
        classify_10d[i,0] = 1
        if classify_10d[i,0] == test_data2[i,10]:
            tp_10d_PG = tp_10d_PG + 1
        else:
            fp_10d_PG = fp_10d_PG + 1
    else:
        classify_10d[i,0] = 0
        if classify_10d[i,0] == test_data2[i,10]:
            tn_10d_PG = tn_10d_PG + 1
        else:
            fn_10d_PG = fn_10d_PG + 1


""" ======================================= Test Crab K-Nearest Neighbors Model ========================================== """
Z_crab = clffinal_crab.predict(test_data1fortest)

tp_crab_KNN = 0
fp_crab_KNN = 0
fn_crab_KNN = 0
tn_crab_KNN = 0
for i in range(len(pos1_crab)):
    if Z_crab[i] == 1:
        if Z_crab[i] == test_data1[i,7]:
            tp_crab_KNN = tp_crab_KNN + 1
        else:
            fp_crab_KNN = fp_crab_KNN + 1
    else:
        if Z_crab[i] == test_data1[i,7]:
            tn_crab_KNN = tn_crab_KNN + 1
        else:
            fn_crab_KNN = fn_crab_KNN + 1


""" ======================================= Test 10D K-Nearest Neighbors Model ========================================== """
Z_10d = clffinal_10d.predict(test_data2fortest)

tp_10d_KNN = 0
fp_10d_KNN = 0
fn_10d_KNN = 0
tn_10d_KNN = 0
for i in range(len(pos1_10d)):
    if Z_10d[i] == 1:
        if Z_10d[i] == test_data2[i,10]:
            tp_10d_KNN = tp_10d_KNN + 1
        else:
            fp_10d_KNN = fp_10d_KNN + 1
    else:
        if Z_10d[i] == test_data2[i,10]:
            tn_10d_KNN = tn_10d_KNN + 1
        else:
            fn_10d_KNN = fn_10d_KNN + 1


""" ======================================= Plot 1========================================== """
fig1 = plt.figure()
ax11 = fig1.add_subplot(121)
ax12 = fig1.add_subplot(122)
ax11.set_title('train data')
ax12.set_title('test data')
ax11.set_xlabel('4th feature')
ax11.set_ylabel('5th feature')
ax12.set_xlabel('4th feature')
ax12.set_ylabel('5th feature')

ax11.scatter(train_data2T[:,4], train_data2T[:,5]) 
ax11.scatter(train_data2F[:,4], train_data2F[:,5], c='r') 
ax12.scatter(test_data2T[:,4], test_data2T[:,5]) 
ax12.scatter(test_data2F[:,4], test_data2F[:,5], c='r') 


""" ======================================= Plot 2========================================== """
fig2 = plt.figure(figsize=(10,10))
ax21 = fig2.add_subplot(121)
ax22 = fig2.add_subplot(122)
ax21.set_title('train data')
ax22.set_title('test data')
classify_10d_train = plot1gp(train_data2fortrain)
fpr_train,tpr_train,threshold_train = roc_curve(train_data2forcls, classify_10d_train)
roc_auc_train = auc(fpr_train,tpr_train)
fpr_test,tpr_test,threshold_test = roc_curve(test_data2forcls, classify_10d)
roc_auc_test = auc(fpr_test,tpr_test)

lw = 2
ax21.plot(fpr_train, tpr_train, color='darkorange',
        lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_train)
ax21.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
ax21.set_xlim([0.0, 1.0])
ax21.set_ylim([0.0, 1.05])
ax21.set_xlabel('False Positive Rate')
ax21.set_ylabel('True Positive Rate')
ax21.set_title('train data')
ax21.legend(loc="lower right")

ax22.plot(fpr_test, tpr_test, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_test)
ax22.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
ax22.set_xlim([0.0, 1.0])
ax22.set_ylim([0.0, 1.05])
ax22.set_xlabel('False Positive Rate')
ax22.set_ylabel('True Positive Rate')
ax22.set_title('test data')
ax22.legend(loc="lower right")


""" ======================================= Plot 3========================================== """
#k_crab, accuracy_crab = plot3_crab(test_data1fortest, test_data1forcls)
k_10d, accuracy_10d = plot3_10d(test_data2fortest, test_data2forcls)
#k_crab_train, accuracy_crab_train = plot3_crab(train_data1fortrain, train_data1forcls)
k_10d_train, accuracy_10d_train = plot3_10d_train(train_data2fortrain, train_data2forcls)

fig3 = plt.figure()
ax31 = fig3.add_subplot(121)
ax32 = fig3.add_subplot(122)
#ax33 = fig3.add_subplot(223)
#ax34 = fig3.add_subplot(224)
ax31.set_title('10d test data')
ax32.set_title('10d train data')
ax31.set_xlabel('k')
ax31.set_ylabel('accuracy')
ax32.set_xlabel('k')
ax32.set_ylabel('accuracy')

ax31.plot(k_10d,accuracy_10d)
ax32.plot(k_10d_train,accuracy_10d_train)
plt.show()




