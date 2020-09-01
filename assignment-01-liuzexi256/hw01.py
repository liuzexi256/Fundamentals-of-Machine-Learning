#-*- coding: utf-8 -*-
"""
File:   hw01.py
Author: Zexi Liu
Date:   
Desc:   
    
"""


""" =======================  Import dependencies ========================== """

import numpy as np
import matplotlib.pyplot as plt
import math
import textwrap
from scipy.stats import norm

plt.close('all') #close any open plots

"""
===============================================================================
===============================================================================
============================ Question 1 =======================================
===============================================================================
===============================================================================
"""
""" ======================  Function definitions ========================== """

def plotData(x1,t1,x2=None,t2=None,x3=None,t3=None,legend=[]):
    '''plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]): Generate a plot of the 
       training data, the true function, and the estimated function'''
    p1 = plt.plot(x1, t1, 'bo') #plot training data
    if(x2 is not None):
        p2 = plt.plot(x2, t2, 'g')
    if(x3 is not None):
        p3 = plt.plot(x3, t3, 'r')

    #add title, legend and axes labels
    plt.ylabel('t')
    plt.xlabel('x')
    
    if(x2 is None):
        plt.legend((p1[0]),legend)
    if(x3 is None):
        plt.legend((p1[0],p2[0]),legend)
    else:
        plt.legend((p1[0],p2[0],p3[0]),legend)
      
def fitdataLS(x,t,M):
    '''fitdataLS(x,t,M): Fit a polynomial of order M to the data (x,t) using LS''' 

    X = np.array([x**m for m in range(M+1)]).T
    w = np.linalg.inv(X.T@X)@X.T@t

    return w

def fitdataIRLS(x,t,M,k):
    '''fitdataIRLS(x,t,M,k): Fit a polynomial of order M to the data (x,t) using IRLS'''

    X = np.array([x**m for m in range(M+1)]).T
    w = np.linalg.inv(X.T@X)@X.T@t
    wprev = 0
    while abs(np.sum(w)-np.sum(wprev)) > 0.0001:
        b = [([0]*len(x)) for i in range(len(x))]
        
        for i in range(len(x)):
            if abs(t[i] - X[i,:]@w) <= k:
                b[i][i] = 1
            else:
                b[i][i] = k/abs(t[i] - X[i,:]@w)
        wprev = w
        w = np.linalg.inv(X.T@b@X)@X.T@b@t

    return w
        

""" ======================  Variable Declaration ========================== """
M = 10 #regression model order
k = 0.03 #Huber M-estimator tuning parameter

""" =======================  Load Training Data ======================= """
train_data = np.load('TrainData.npy')
x1 = train_data[:,0]
t1 = train_data[:,1]

    
""" ========================  Train the Model ============================= """
wLS = fitdataLS(x1,t1,M) 
wIRLS = fitdataIRLS(x1,t1,M,k) 


""" ======================== Load Test Data  and Test the Model =========================== """

"""This is where you should load the testing data set. You shoud NOT re-train the model   """
test_data = np.load('TestData.npy')
x1_test = test_data[:,0]
t1_test = test_data[:,1]

#for plot2
Nfor2 = 20
tyLS = [0 for i in range(Nfor2)]
tyIRLS = [0 for i in range(Nfor2)]

for i in range(Nfor2):
    wLS_test = fitdataLS(x1,t1,i)
    wIRLS_test = fitdataIRLS(x1,t1,i,k)
    X_test = np.array([x1_test**m for m in range(i+1)]).T
    hLS = 0
    hIRLS = 0
    for j in range(len(x1_test)):
        hLS = hLS + abs(t1_test[j] - X_test[j,:]@wLS_test)
        hIRLS = hIRLS + abs(t1_test[j] - X_test[j,:]@wIRLS_test)

    tyIRLS[i] = hIRLS
    tyLS[i] = hLS

#for plot3
Nfor3 = 20
tyIRLS3 = [0 for i in range(Nfor3)]
X_test3 = np.array([x1_test**m for m in range(M+1)]).T
for i in range(Nfor3):
    I = 0.01*i + 0.001
    wIRLS_test3 = fitdataIRLS(x1,t1,M,I)
    hIRLS = 0
    for j in range(len(x1_test)):
        hIRLS = hIRLS + abs(t1_test[j] - X_test3[j,:]@wIRLS_test3)
    tyIRLS3[i] = hIRLS


""" ========================  Plot Results ============================== """
plt.figure(num = 1,figsize = (5,5))

ax1 = plt.subplot(3,1,1)
ax2 = plt.subplot(3,1,2)
ax3 = plt.subplot(3,1,3)

#plot1
plt.sca(ax1)
xrange = np.arange(-4.5,4.5,0.001)  #get equally spaced points in the xrange
XLS = np.array([xrange**m for m in range(wLS.size)]).T
XIRLS = np.array([xrange**m for m in range(wIRLS.size)]).T

estyLS = XLS@wLS #compute the predicted value
estyIRLS = XIRLS@wIRLS
plotData(x1,t1,xrange,estyLS,xrange,estyIRLS,legend = ['Training Data','least squares solution','huber M-estimator solution'])
plt.xlim(-4.5, 4.5)
plt.ylim(-2, 2)


#plot2
plt.sca(ax2)
Mx = [i for i in range(Nfor2)]
plt.scatter(Mx,tyLS, label = 'least squares')
plt.scatter(Mx,tyIRLS, label = 'Huber M-estimator')
plt.ylabel('sum(|t-y|)')
plt.xlabel('M')
plt.xlim(0, 22)
#plt.ylim(0, 500)
plt.legend()


#plot3
plt.sca(ax3)
kx = [0.01*i + 0.001 for i in range(Nfor3)]
plt.scatter(kx,tyIRLS3, label = 'Huber M-estimator')
plt.ylabel('sum(|t-y|)')
plt.xlabel('k')
plt.legend()


plt.show()
