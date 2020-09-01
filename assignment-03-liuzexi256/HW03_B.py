"""
File:   hw03.py
Author: Zexi Liu
Date:   
Desc:   Assignment 03B 
    
"""


""" =======================  Import dependencies ========================== """
import numpy as np
from sklearn.decomposition import PCA
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import datasets 
""" =======================  1 PCA  ========================== """
X = np.array([[2,3,3,4,5,7],[2,4,5,5,6,8]])
X_sm = np.array([[-2,-1,-1,0,1,3],[-3,-1,0,0,1,3]])
XTX = np.dot(X.T,X)
XXT = np.dot(X,X.T)
XXTsm = np.dot(X_sm,X_sm.T)
A = np.linalg.eig(XTX)
B = np.linalg.eig(XXT)
C = np.linalg.eig((1/6)*XXTsm)



pca = PCA(n_components=1)
X_spca = pca.fit_transform(X.T)


""" =======================  2 EM  ========================== """
def EM(Y,NumberOfComponents,Iterations,u,p,q):

    temp1 = 0
    temp2 = 0
    temp3 = 0
    NumberIterations = 1
    mu = np.zeros( (1,NumberOfComponents) )
    for  j in range(10):
        mu[0,j] = (u*(p**Y[j])*((1-p)**(1-Y[j])))/((u*(p**Y[j])*((1-p)**(1-Y[j]))) + (1-u)*(q**Y[j])*((1-q)**(1-Y[j])))
    
    while NumberIterations <= Iterations:
        uNew = np.mean(mu)
        temp1 = np.sum(mu*Y)
        temp2 = np.sum([1 - x for x in mu]*Y)
        temp3 = np.sum([1 - x for x in mu])
        pNew = temp1/np.sum(mu)
        qNew = temp2/temp3
        for  j in range(10):
            mu[0,j] = (uNew*(pNew**Y[j])*((1-pNew)**(1-Y[j])))/((uNew*(pNew**Y[j])*((1-pNew)**(1-Y[j]))) + (1-uNew)*(qNew**Y[j])*((1-qNew)**(1-Y[j])))
        print('NumberIterations = ', NumberIterations)
        print('mu=',mu)
        print('pi=',uNew)
        print('q=',qNew)
        print('p=',pNew)
        NumberIterations += 1
    return mu, uNew, pNew, qNew

Y = np.array([1,1,0,1,0,0,1,0,1,1])
mu1, u1, p1, q1 = EM(Y,10,20,0.4,0.6,0.7)
mu2, u2, p2, q2 = EM(Y,10,20,0.5,0.5,0.5)
print('mu1=',mu1)
print('pi1=',u1)
print('q1=',q1)
print('p1=',p1)
print('mu2=',mu2)
print('pi2=',u2)
print('q2=',q2)
print('p2=',p2)
print('good')



        