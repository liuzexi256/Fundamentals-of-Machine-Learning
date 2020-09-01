# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 09:53:05 2019

@author: jpeeples
"""
import torch
import numpy as np
import copy
import matplotlib.pyplot as plt

def train_model(model,epochs,loss_fxn,lr_rate,training_data,labels,print_epochs):
    
    optimizer = torch.optim.Adam(model.parameters(),lr=lr_rate)
    model.train()
    Error_track = np.zeros(epochs)
    error_check = np.inf
    epoch_plt = np.arange(0,epochs)
    loss_plt = []
    for epoch in range(0,epochs):
        
        #Forward pass through model
        output = torch.flatten(model(training_data))
        
        #Compute loss
        error = loss_fxn(output,labels)
        loss_plt.append(error)
        #Keep track of Error
        Error_track[epoch] = error.detach().numpy()
        
        #Update weights with backprop
        error.backward()
        optimizer.step()
        
        #Save best model
        if error < error_check:
            error_check = error
            best_model_wts = copy.deepcopy(model.state_dict())
        if epoch % print_epochs == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(
                epoch, error.item()))
        
    print()
    plt.figure()
    plt.plot(epoch_plt,loss_plt)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    #Return model weights and training loss
    return best_model_wts,Error_track