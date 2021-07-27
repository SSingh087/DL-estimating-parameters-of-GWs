
import numpy as np
from tqdm import tqdm
#---Data reading and writing---------------
import csv
import h5py
import pandas as pd


hf= h5py.File('/../data.h5', 'r')
signal_gw= np.array(hf[list(hf.keys())[0]])
hf.close()

labels=pd.read_csv('/../labels.csv')
labels.shape

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split

data=signal_gw
print(data.shape)
data=data.reshape((data.shape[0],data.shape[1],1))
print(data.shape)

labels=pd.read_csv('gdrive/My Drive/GW data/labels_mixed.csv')
labels=labels.to_numpy(dtype=int,copy=True)

X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=1, test_size=0.2)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

model = Sequential()
..
..
..
..

#file 
