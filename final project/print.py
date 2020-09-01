from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle

def load_pkl(fname):
    with open(fname,'rb') as f:
        return pickle.load(f)

train_dataset = load_pkl('train_data.pkl')
for m in range(64):
    figure1 = plt.figure(1)
    for n in range(100):
        figure1.add_subplot(10,10,n+1)
        data = np.array(train_dataset[(20+m)*100+n])
        new_im = Image.fromarray(data)
        n = n + 1
        plt.imshow(new_im)
        
        print(n)
    plt.show() 