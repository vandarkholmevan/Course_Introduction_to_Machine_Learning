import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn import manifold
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class K_means():
    def __init__(self, dataname, K_num, epoch):
        self.dataname = dataname
        self.datapath = os.path.join(os.path.dirname(__file__), self.dataname)
        self.K_num    = K_num
        self.epoch    = epoch
        self.enc      = LabelEncoder()
        self.scaler   = MinMaxScaler()
    
    def initialize(self):
        self.counter_arr  = np.zeros([self.X_values.shape[0], self.K_num])
        self.category_arr = np.zeros([1, self.X_values.shape[0]])
        self.sum_arr      = np.zeros([self.K_num, self.X_values.shape[1]])
        self.landmark     = np.array([[np.random.rand()] * self.X_values.shape[1] for i in range(self.K_num)])
        
    def calculate_distance(self, data, landmark):
        reg1_dis = data - landmark
        reg2_dis = np.sqrt(np.sum(reg1_dis ** 2))
        return reg2_dis
    
    def preprocess(self, X, Y):
        Y = self.enc.fit_transform(Y)
        X = self.scaler.fit_transform(X)
        return X, Y
        
    def plot(self, X_values, category_arr, epoch):
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=1001)
        X_tsne = tsne.fit_transform(X_values)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=category_arr)
        plt.savefig('./img/img' + str(epoch) +'.png')

    def train(self):
        self.path = os.path.join(self.datapath, self.dataname)
        self.df   = pd.read_csv(self.datapath)
        self.X_values, self.Y_values = self.preprocess(self.df.values[:, :-1], self.df.values[:, -1])
        
        self.initialize()
        for ep in range(self.epoch):
            for data_idx in range(self.X_values.shape[0]):
                for lmark_idx in range(self.K_num):
                    self.counter_arr[data_idx, lmark_idx] = self.calculate_distance(self.X_values[data_idx], self.landmark[lmark_idx])
            
            self.category_arr = np.argmax(self.counter_arr, axis=1).T
            for lmark_idx in range(self.K_num):
                if self.X_values[(self.category_arr==lmark_idx).T, :].shape[0] == 0:
                    self.landmark[lmark_idx, :] = np.array([np.random.rand()] * self.X_values.shape[1])
                else :
                    self.landmark[lmark_idx, :] = np.mean(self.X_values[(self.category_arr==lmark_idx).T, :], axis=0)
                
            self.plot(self.X_values, self.category_arr, ep)
