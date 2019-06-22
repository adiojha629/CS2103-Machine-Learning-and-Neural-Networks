# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 11:16:00 2019

@author: Adi
"""
import numpy as np
from PerceptronClass import Perceptron
#Debugging Functions
def printArray(x):
    for i in range(len(x)):
        for a in range(len(x[i])):
            print("Row " + str(i),end = " ")
            print("Column " + str(a),end = " ")
            print(x[i][a])


class NueralNetwork:
    def __init__(self,training_data,network_setup):
        self.train_data = training_data
        '''
        Train_data is a list of dictionaries. Each Dictionary has two keys:
            keys = 'data' and 'target'
            The data values are an array of inputs(same for all dicts), 
            and the target values are a single number (0 or 1) 
        '''
        self.net_setup = network_setup
        '''
        net_setup is a list of values. Each value is how many perceptrons are 
            on each level
            net_setup.length = number of inner network levels, excluding 
            data input level and the output level
            ie net_setup = [2,3,2]: 3 inner levels; 2 perceptrons on level 1 etc
        '''
        self.network = []
        #^^^holds a 2D array with perceptrons organized like net_setup dictates
        
        for i in range(len(self.net_setup)):
            level = []
            for num in range(self.net_setup[i]):
                if (i == 0):
                    p = Perceptron(len(self.train_data[0]['data']))
                else:
                    p = Perceptron(self.net_setup[i-1])
                level.append(p)
            self.network.append(level)
        #printArray(self.network)
    
    def fwdProp(self,p):
        enterData = []
        enterData.append(p['data'])
        for i in range(len(self.network)):
            tempData = []
            for a in range(len(self.network[i])):
                tempData.append(self.network[i][a].getPred(enterData[i]))
            enterData.append(tempData)
            
        outputLayer = Perceptron(self.net_setup[-1])
        return outputLayer.getPred(enterData[-1])
    
        

data = [{'data': [3,1.5], 'target': 1}, {'data':[2,1], 'target': 0},
        {'data': [4,1.5], 'target': 1}, {'data': [3,1], 'target': 0},
        {'data': [3.5,.5], 'target': 1}, {'data': [2,.5], 'target': 0},
        {'data': [5.5,1], 'target': 1}, {'data': [1,1], 'target': 0}]
n = NueralNetwork(data, [2,3,2])
print(n.fwdProp(data[0]))
          
            
