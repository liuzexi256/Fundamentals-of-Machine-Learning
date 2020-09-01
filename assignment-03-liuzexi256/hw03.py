# -*- coding: utf-8 -*-
"""
File:   hw03.py
Author: Zexi Liu
Date:   
Desc:   Assignment 03B,03C 
    
"""

"""
====================================================
================ Import Packages ===================
====================================================
"""
import sys
import numpy as np
from sklearn.decomposition import PCA
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import datasets 
from PIL import Image
from skimage.color import rgb2gray
import skimage.filters as filt

"""
====================================================
================ Define Functions ==================
====================================================
"""
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

def process_image(in_fname,out_fname,debug=False):

    # load image
    x_in = np.array(Image.open(in_fname))

    # convert to grayscale
    x_gray = 1.0-rgb2gray(x_in)

    if debug:
        plt.figure(1)
        plt.imshow(x_gray)
        plt.title('original grayscale image')
        plt.show()

    # threshold to convert to binary
    thresh = filt.threshold_minimum(x_gray)
    fg = x_gray > thresh

    if debug:
        plt.figure(2)
        plt.imshow(fg)
        plt.title('binarized image')
        plt.show()

    # find bounds
    nz_r,nz_c = fg.nonzero()
    n_r,n_c = fg.shape
    l,r = max(0,min(nz_c)-1),min(n_c-1,max(nz_c)+1)+1
    t,b = max(0,min(nz_r)-1),min(n_r-1,max(nz_r)+1)+1

    # extract window
    win = fg[t:b,l:r]

    if debug:
        plt.figure(3)
        plt.imshow(win)
        plt.title('windowed image')
        plt.show()

    # resize so largest dim is 48 pixels 
    max_dim = max(win.shape)
    new_r = int(round(win.shape[0]/max_dim*48))
    new_c = int(round(win.shape[1]/max_dim*48))

    win_img = Image.fromarray(win.astype(np.uint8)*255)
    resize_img = win_img.resize((new_c,new_r))
    resize_win = np.array(resize_img).astype(bool)

    # embed into output array with 1 pixel border
    out_win = np.zeros((resize_win.shape[0]+2,resize_win.shape[1]+2),dtype=bool)
    out_win[1:-1,1:-1] = resize_win

    if debug:
        plt.figure(4)
        plt.imshow(out_win,cmap='Greys')
        plt.title('resized windowed image')
        plt.show()

    #save out result as numpy array
    np.save(out_fname,out_win)
    return out_win

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
Y = np.array([1,1,0,1,0,0,1,0,1,1])
mu1, u1, p1, q1 = EM(Y,10,20,0.4,0.6,0.7)
mu2, u2, p2, q2 = EM(Y,10,20,0.5,0.5,0.5)
"""
====================================================
========= Generate Features and Labels =============
====================================================
"""
'''
if __name__ == '__main__':

    # To not call from command line, comment the following code block and use example below 
    # to use command line, call: python hw03.py K.jpg output

    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print('usage: {} <in_filename> <out_filename> (--debug)'.format(sys.argv[0]))
        sys.exit(0)
    
    in_fname = sys.argv[1]
    out_fname = sys.argv[2]

    if len(sys.argv) == 4:
        debug = sys.argv[3] == '--debug'
    else:
        debug = False
'''

#    #e.g. use
#    process_image('C:/Desktop/K.jpg','C:/Desktop/output.npy',debug=True)
output = []
labels = []
#output = output.tolist()
for n in range(8):
    if n == 0:    
        for m in range(10):
            filename = 'a'+str(m)+'.png'
            singleout = process_image(filename,'output.npy',debug=False)
            singleout = singleout.tolist()
            output.append(singleout)
            labels.append(1)
    if n == 1:
        for m in range(10):
            filename = 'b'+str(m)+'.png'
            singleout = process_image(filename,'output.npy',debug=False)
            singleout = singleout.tolist()
            output.append(singleout)
            labels.append(2)
    if n == 2:
        for m in range(10):
            filename = 'c'+str(m)+'.png'
            singleout = process_image(filename,'output.npy',debug=False)
            singleout = singleout.tolist()
            output.append(singleout)
            labels.append(3)
    if n == 3:
        for m in range(10):
            filename = 'd'+str(m)+'.png'
            singleout = process_image(filename,'output.npy',debug=False)
            singleout = singleout.tolist()
            output.append(singleout)
            labels.append(4)
    if n == 4:
        for m in range(10):
            filename = 'h'+str(m)+'.png'
            singleout = process_image(filename,'output.npy',debug=False)
            singleout = singleout.tolist()
            output.append(singleout)
            labels.append(5)
    if n == 5:
        for m in range(10):
            filename = 'i'+str(m)+'.png'
            singleout = process_image(filename,'output.npy',debug=False)
            singleout = singleout.tolist()
            output.append(singleout)
            labels.append(6)
    if n == 6:
        for m in range(10):
            filename = 'j'+str(m)+'.png'
            singleout = process_image(filename,'output.npy',debug=False)
            singleout = singleout.tolist()
            output.append(singleout)
            labels.append(7)
    if n == 7:
        for m in range(10):
            filename = 'k'+str(m)+'.png'
            singleout = process_image(filename,'output.npy',debug=False)
            singleout = singleout.tolist()
            output.append(singleout)
            labels.append(8)
data = np.array(output)
labels = np.array(labels)
np.save('data.npy',data)
np.save('labels.npy',labels)
D = np.load('data.npy',allow_pickle=True)
L = np.load('labels.npy')