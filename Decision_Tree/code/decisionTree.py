#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 13:42:56 2019

@author: zhaocheng_du
"""

import numpy as np
from collections import Counter
import sys

def readFile(filename):
    '''
    inpit the filename
    return the data and its column
    '''
    data = []
    inf = open(filename)
    for i in inf:
        data.append(i.strip('\n').split('\t'))
    return np.array(data[1:].copy()).astype(object), np.array(data.pop(0)).astype(object)

def Entropy(Y):
    '''
    take in array format
    '''
    Y_sum = len(Y)
    Y_cnt = list(Counter(Y.tolist()).values())
    Y_etp = sum([- i / Y_sum * np.log2(i / Y_sum) for i in Y_cnt])
    return Y_etp

def muInfo(Y, X):
    '''
    X,Y in array format, return the mutual information. 
    Y is label, X is input.
    '''
    X_len    = len(X)
    uni_elem = np.unique(X)
    mu_info  = Entropy(Y)
    for i in uni_elem:
        mu_info  -= (np.count_nonzero(X == i) / X_len) * Entropy(Y[X == i])
    return mu_info

def errRate(yPred, yReal):
    '''
    take in array format
    '''
    return np.count_nonzero(yPred != yReal) / len(yPred)

class Node():
    def __init__(self, X, Y, depth, idx_li, Ncol, maxDepth, father = None):
        self.left   = None
        self.right  = None
        self.father = father
        self.llable = None
        self.rlable = None
        self.X     = X # the X data stored in this node
        self.Y     = Y # the Y data stored in this node
        try:
            self.label = Counter(Y).most_common(1)[0][0] # majority vote
        except:
            self.label = Y[0]
        self.depth = depth
        self.idx   = None # the present column's idx not the absolute idx
        self.Xcol  = idx_li
        self.Ncol  = Ncol # record the names of all columns
        self.idNcol = None # record the split name string
        self.maxDepth = maxDepth
                      
    def chooseAtt(self):
        '''
        return the attribution's index
        '''
        mu_li = []
        for i in self.Xcol:
            mu_li.append(muInfo(self.Y, self.X[:, i]))
        return self.Xcol[mu_li.index(max(mu_li))]
    
    def splitNode(self):
        '''
        split the node into several sub-tree
        '''
        idx = self.chooseAtt()
        self.idx    = idx
        self.idNcol = self.Ncol[idx]
        ele = self.X[:, idx]
        uni = np.unique(ele)
        newidx_li = self.Xcol.copy()
        newidx_li.remove(idx)
        if len(uni) == 2:
            if self.Y[ele == uni[0]].any():
                self.left   = Node(self.X[ele == uni[0], :], self.Y[ele == uni[0]], 
                               self.depth + 1, newidx_li, self.Ncol, self.maxDepth, self)
                self.llable = uni[0]
            if self.Y[ele == uni[1]].any():
                self.right  = Node(self.X[ele == uni[1], :], self.Y[ele == uni[1]], 
                               self.depth + 1, newidx_li, self.Ncol, self.maxDepth, self)
                self.rlable = uni[1]
        elif len(uni) == 1:
            self.left   = Node(self.X[ele == uni[0], :], self.Y[ele == uni[0]], 
                               self.depth + 1, newidx_li, self.Ncol, self.maxDepth, self)
            self.llable = uni[0]
    
    def train(self):
        '''
        Train a decision tree
        '''
        if  not(self.Xcol):
            print("[WARNING]: This depth you assigned is bigger than the number of attributes!")
            return
        if (len(np.unique(self.Y)) == 1) or (self.depth == self.maxDepth):
            return 
        self.splitNode()
        if self.left:
            self.left.train()
        if self.right:
            self.right.train()
        return
    
    def predictRow(self, X_row): 
        if (not self.left) and (not self.right):
            return self.label
        elif X_row[self.idx] == self.llable:
            return self.left.predictRow(X_row)
        elif X_row[self.idx] == self.rlable:
            return self.right.predictRow(X_row)
        elif (not self.left) or (not self.right):
            return self.label
        else:
            print(X_row[self.idx])
            raise Exception('no label')
    
    def predict(self, testX):
        outcome = []
        for row in testX:
            outcome.append(self.predictRow(row))
        return np.array(outcome)
   
    def nodePrint(self, pos=None):
        cate = np.unique(self.Y)
        cnum = np.array([np.count_nonzero(self.Y == i) for i in cate])
        if self.depth == 1:
            print('[', end = '')
            for i in range(len(cate)):
                print(cate[i] + str(cnum[i]), end = '')
                print('/', end = '')
            print(']')
        else:
            for i in range(self.depth-1):
                print('|\t', end = '')
            lable = self.father.llable if pos == 'left' else self.father.rlable
            print(self.father.idNcol + ' = ' + lable + ': ', end = '')
            print('[', end = '')
            for i in range(len(cate)):
                print(cate[i] + ' ' + str(cnum[i]), end = '')
                print('/', end = '')
            print(']', end = ' ')
            print(self.label)
                
    def treePrint(self, pos=None):
        if self.maxDepth == 1:
            self.nodePrint()
        else:
            self.nodePrint(pos)
            if self.left:
                self.left.treePrint('left')
            if self.right:
                self.right.treePrint('right')

        
if __name__ == '__main__':
    TRAIN_IN = sys.argv[1]
    TEST_IN  = sys.argv[2]
    STOP_PNT = sys.argv[3]
    TRAIN_OU = sys.argv[4]
    TEST_OU  = sys.argv[5]
    METRIC   = sys.argv[6]
    
    print(STOP_PNT)

    tni_data, tni_col = readFile(TRAIN_IN)
    tsi_data, tsi_col = readFile(TEST_IN,)
    
    root = Node(tni_data[:, :-1], tni_data[:, -1], 1, 
                [i for i in range(tni_data[:-1].shape[1] - 1)], 
                tni_col, int(STOP_PNT) + 1)
    root.train()
    
    trainOutput = root.predict(tni_data)
    testOutput  = root.predict(tsi_data)
    
    trainErr = errRate(trainOutput, tni_data[:, -1])
    testErr  = errRate(testOutput, tsi_data[:, -1])
    
    with open(TRAIN_OU, 'w') as nof:
        for i in trainOutput:   
            nof.write(str(i) + '\n')
    with open(TEST_OU, 'w') as sof:
        for i in testOutput:   
            sof.write(str(i) + '\n')
    with open(METRIC, 'w') as mof:
        mof.write('error(train): ' + str(trainErr) + '\n')
        mof.write('error(test): ' + str(testErr) + '\n')
    root.treePrint()
    
