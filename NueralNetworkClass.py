# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 11:16:00 2019

@author: Adi
"""

from matplotlib import pyplot as plt
import numpy as np
from PerceptronClass import Perceptron
import math
import random
#Debugging Functions
def printArray(x):
    for i in range(len(x)):
        for a in range(len(x[i])):
            print("Row " + str(i),end = " ")
            print("Column " + str(a),end = " ")
            print(x[i][a])
        print("",end = "\n")
def printArray3D(x):
    for i in range(len(x)):
        for a in range(len(x[i])):
            for b in range(len(x[i][a])):
                print("X " + str(i),end = " ")
                print("Y " + str(a),end = " ")
                print("Z " + str(b),end = " ")
                print(x[i][a][b])
        print("",end = "\n")


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
        if(len(self.net_setup) == 0):
            self.net_setup.append(len(self.train_data[0]['data']))
        else:
            for i in range(len(self.net_setup)):
                level = []
                for num in range(self.net_setup[i]):
                    if (i == 0):
                        p = Perceptron(len(self.train_data[0]['data']))
                    else:
                        p = Perceptron(self.net_setup[i-1])
                    level.append(p)
                self.network.append(level)
        
        output_node = Perceptron(self.net_setup[-1])
        output_level = []
        output_level.append(output_node)
        self.network.append(output_level)
        
        #find out how many neurons are present;
        self.numofNodes = 0
        for number in self.net_setup:
            self.numofNodes += number
    
        
        #print("This is the Network\n\n")
        #printArray(self.network)
    
    def __str__(self):
        printArray(self.network)
        return "Object Printed"
      
    '''
    def inputData(self,x):
        
            if x has a target then train
            else give me prediction
            
            if x is a list of dictionaries, then add that data to self.trainData
            if x is just a dictionary, then add it to self.trainData
            if x 
            
        
        
        
        if('target' in x.keys()):
            ###
        elif('data' in x.keys()): #This means that we want a prediction
            print("The Prediction is: " + str(self.fwdProp(x['data'])))
        else:
            print("Please format the data into a dictionary)
            '''
            
    
    def fwdProp(self,p): #This Works
        enterData = []
        #print(enterData)
        enterData.append(p['data'])
        #print(enterData)
        '''
        node = self.network[0][0]
        print("Number of Weights equals "+ str(len(node.getWeights())))
        print("Number of variables equals "+ str(len(enterData[0])))
        print(enterData[0])
        '''
        for i in range(len(self.network)):
            tempData = []
            for a in range(len(self.network[i])):
                node = self.network[i][a]
                #print(node.getWeights())
                tempData.append(node.getPred(enterData[i]))
            enterData.append(tempData)
        
        #print("\n\nThis is Enter Data\n")
        #printArray(enterData)
        return enterData[-1][0]
    
    def cost(self,p): #This works
        pred = self.fwdProp(p)
        target_value = p['target']
       # print(type(pred))
        return np.square(target_value - pred)
        
    def getAllWeights(self):#This works
        all_weights = []
        for level in self.network:# Each array in self.network holds all the nodes in a given level, so the reference to this array is level
            weightsforlevel = []
            for node in level:#Each object in level is a node
                weightsforlevel.append(node.getWeights())
            all_weights.append(weightsforlevel)
        return all_weights
    
    
    def backProp(self,p):
        H = 1e-12
        #print("H value is " + str(H))
        learningRate = 0.03 /(.1 * self.numofNodes)
        #print("H value is " + str(learningRate))
        d_costs = []
        #print("Before Training, this is d_costs")
        #print(d_costs)
        #self.enterBreak()
        network_weights = self.getAllWeights()
        #print("\nNetwork_Weights")
        #print(network_weights)
        for a in range(len(network_weights)):#Each array represents a list of weights for a inner level, hence the variable name level
            level = network_weights[a]
            temp_levelarray = []
            for counter in range(len(level)):
                old_weights = level[counter].copy()
                temp_nodearray = []
                for i in range(len(old_weights)):# for iterating through each weight
                    #self.enterBreak()
                    #print("a is "+ str(a))
                    #print("counter is " + str(counter))
                    #print("i is " + str(i))
                    #print("level is "+ str(level))
                    #print("temp_levelarray is " +str(temp_levelarray))
                    #print("old_weights is " + str(old_weights))
                    #print("temp_nodearray is " + str(temp_nodearray))
                    #self.enterBreak()
                    
                    node = self.network[a][counter]
                    #print("Current Node is "+ str(node))
                    
                    
                    wTemp_PlusH = old_weights.copy()
                    wTemp_PlusH[i] += H
                    #print("weights plus H is "+ str(wTemp_PlusH))
                    wTemp_MinusH = old_weights.copy()
                    wTemp_MinusH[i] -= H
                    #print("weights minus H is "+ str(wTemp_MinusH))
                    #print(node.getWeights())
                    #self.enterBreak()
                    #print("Cost Before " + str(self.cost(p)))
                    node.changeWeights(wTemp_PlusH)
                    #print("Current Node should have Plus H "+ str(node))
                    #print(node.getWeights())
                    cost_PlusH = self.cost(p)
                    
                    #self.enterBreak()
                    
                    #print("cost_PlusH " + str(cost_PlusH))
                    node.changeWeights(wTemp_MinusH)
                   # print("Current Node should have Minus H "+ str(node))
                    #print(node.getWeights())
                    cost_MinusH = self.cost(p)
                                        
                    #print("cost_MinusH " + str(cost_MinusH))
                    node.changeWeights(old_weights)#Set weights back to original values
                    #print(node.getWeights(), end = "\n\n")
                    #print("Current Node should be normal "+ str(node))
                    #self.enterBreak()
                    derivative = (cost_PlusH-cost_MinusH)/(2*H)
                    #print("Derivative is "+ str(derivative),end = "\n\n")
                    temp_nodearray.append(derivative * learningRate)
                temp_levelarray.append(temp_nodearray)
            d_costs.append(temp_levelarray)
        #print("This is d_costs\n\n")
        #printArray(d_costs)
        return d_costs
    
    def trainNetwork(self):#This is working
        for num in range(1000):
            ri = np.random.randint(len(self.train_data))
            #print(ri)
            p = self.train_data[ri]
            d_costs = self.backProp(p)
            for i in range(len(d_costs)):
                for a in range(len(d_costs[i])):
                    self.network[i][a].trainWeights(d_costs[i][a])
                    #print("Hello")
            
        return "Network Trained"
     
    def accuracy(self):
        numCorrect = 0
        for i in range(100000):
            ri = np.random.randint(len(self.train_data))
            p = self.train_data[ri]
            pred = self.fwdProp(p)
            target = p['target']
            if( (target == 1 and pred > .5) or (target == 0 and pred < .5)):
                numCorrect += 1
            
        return numCorrect/100000
        
    #Debugging Methods
    def getNode(self,level,number):
        return self.network[level][number]
    def enterBreak(self):
        user = ""
        return user
        
'''      
        

data = [{'data': [3,3], 'target': 1}, {'data':[2,1], 'target': 0},
        {'data': [4,5], 'target': 1}, {'data': [2.5,1], 'target': 0},
        {'data': [3,5], 'target': 1}, {'data': [2,.5], 'target': 0},
        {'data': [5,7], 'target': 1}, {'data': [1,1], 'target': 0}]
def action():
    for point in data:
        x = point['data'][0]
        y = point['data'][1]
        label = point['target']
        color = 'r' #if the target is 1
        if(label == 0):
            color = 'b' #if the target is 0
        plt.scatter(x,y,c = color)
        
action()


'''
'''
xorData = []

for i in range(100):
    for a in range(100):
        if(i == a):
            xorData.append({'data':[i,a],'target': 0})
        else:
            xorData.append({'data':[i,a],'target': 1})
'''
'''

#Generating Circular Data
data = []
for i in range(250):
    # radius of the circle
    circle_r = 1
    # center of the circle (x, y)
    circle_x = 20
    circle_y = 20

    # random angle
    alpha = 2 * math.pi * random.random()
    # random radius
    r = circle_r * math.sqrt(np.random.randint(0,10))
    # calculating coordinates
    x = r * math.cos(alpha) + circle_x
    y = r * math.sin(alpha) + circle_y
    data.append({'data':[x,y],'target': 0})
    r = 2*circle_r * math.sqrt(np.random.randint(20,30))
    x = r * math.cos(alpha) + circle_x
    y = r * math.sin(alpha) + circle_y
    data.append({'data':[x,y],'target': 1})
'''
'''
#clearly separable data
data = []
for i in range(10):
    x = np.random.randint(0,5)
    y = np.random.randint(0,5)
    data.append({'data':[x,y],'target':1})
    x = np.random.randint(10,15)
    y = np.random.randint(10,15)
    data.append({'data':[x,y],'target':0})
    
     
            

n = NueralNetwork(data, [3])
#point = data[0]

n.trainNetwork()
#print(n.getAllWeights())

print(n.accuracy())

#print(n.fwdProp(point))
#print(n.cost(point))




user = input("Continue?")
while(user != "exit"):
    ri = np.random.randint(len(n.train_data))
    p = data[ri]
    pred = n.fwdProp(p)
    target = p['target']
    print("Data Point " + str(p))
    print("Prediction " + str(pred))
    user = input("Continue?")
    if(user == "train"):
        n.trainNetwork()

'''
'''
print("Before Network was trained, accuracy: " + str(n.accuracy()))
n.trainNetwork()
print("After Network was trained, accuracy: " + str(n.accuracy()))
'''
'''
mystery_flower = {'data': [4.5,1]}

pred_before = n.fwdProp(mystery_flower)
print(n.trainNetwork())

print("Before Training, the prediction is "+ str(pred_before))
print("After Training, the prediction is "+ str(n.fwdProp(mystery_flower)))
#n.backProp(data[0])
'''
#print(n)


#printArray3D(n.getAllWeights())
#print(n.fwdProp(data[0]))
#print(bool(n.cost(data[0]) == np.square(data[0]['target'] - n.fwdProp(data[0]))))
'''
'''
'''
print("Prediction is " + str(n.fwdProp(data[0])))
print("Cost is " + str(n.cost(data[0])))
print("Network Weights:\n")
printArray3D(n.getAllWeights())
print("d_costs:\n")
printArray3D(n.backProp(data[0]))
'''

#print(data[0]['data'])
#print(n.fwdProp(data[0]))

#print(n.getNode(0,0).getWeights())





#n.fwdProp(mystery_flower)
#n.cost(data[0])

'''
'''
'''
mystery_flower = {'data': [4.5,1]}
mystery_flower = {'data':[2,1], 'target': 0}
pred_before = n.fwdProp(mystery_flower)
print(n.trainNetwork())

print("Before Training, the prediction is "+ str(pred_before))
print("After Training, the prediction is "+ str(n.fwdProp(mystery_flower)))
'''
'''
#printArray3D(n.backProp(data[0]))

#print(n.fwdProp(data[0]))
#printArray(n.getAllWeights())
      
            
'''