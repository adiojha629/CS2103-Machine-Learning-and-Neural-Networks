# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 10:59:54 2019

@author: Adi
"""
import numpy as np

class Perceptron:
    def __init__(self, numberofInputs):
        self.weights = []
        for i in range(numberofInputs+1):#Plus 1 so that the last weight is the bias
            self.weights.append(np.random.randn())
        #print(self.weights)
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def getPred(self,data):#Data should be an array
        if(len(data) == (len(self.weights) - 1)):
            pred = 0
            for i in range(len(data)):
                pred += self.weights[i] * data[i]
            pred = self.sigmoid(pred + self.weights[-1])
            return pred
        else:
            return False
        
    def getWeights(self):
        return self.weights
    def changeWeighs(self,x):
        self.weigths = x
#p = Perceptron(3)        
#data = [1,2,3]
#print(p.getPred(data))

