import numpy as np
import pickle

def load_pkl(fname):
    with open(fname,'rb') as f:
        return pickle.load(f)

train_dataset = load_pkl('train_data.pkl')
bad_data = np.loadtxt('0-20.txt')
bad_data = bad_data.astype(np.int16)
train_dataset = train_dataset.tolist()
k = 0
for n in bad_data:
    X = train_dataset[n-k-1]
    new_im = Image.fromarray(X)
    plt.imshow(new_im)
    plt.show()
    del train_dataset[n-k-1]
    k = k + 1