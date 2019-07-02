# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 19:20:55 2019

@author: Adi Ojha
"""
#Network Testing Arena

from NueralNetworkClass import NueralNetwork
from matplotlib import pyplot as plt
import numpy as np

def stepBystep(network):
    user = input("Continue?")
    while(user != "exit"):
        ri = np.random.randint(len(network.train_data))
        p = network.train_data[ri]
        pred = network.fwdProp(p)
        print("Data Point " + str(p))
        print("Prediction " + str(pred))
        user = input("Continue?")
        if(user == "train"):
            network.trainNetwork()
def lighten_color(color, amount):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


data = [{'data': [3,3], 'target': 1}, {'data':[2,1], 'target': 0},
        {'data': [4,5], 'target': 1}, {'data': [2.5,1], 'target': 0},
        {'data': [3,5], 'target': 1}, {'data': [2,.5], 'target': 0},
        {'data': [5,7], 'target': 1}, {'data': [1,1], 'target': 0}]
n = NueralNetwork(data,[4,3])
n.trainNetwork()

for point in data:
        x = point['data'][0]
        y = point['data'][1]
        pred = n.fwdProp(point)
        color = 'r'
        if(pred > .5):
            color = lighten_color('r',.5 - pred)
        elif(pred < .5):
            color = lighten_color('b',0 - pred)
        plt.scatter(x,y,c = color)






plt.show()



n = NueralNetwork(data,[4,3])
'''
stepBystep(n)
print("Accuracy: " + str(n.accuracy()))
n.trainNetwork()
print("Network Trained")
stepBystep(n)
print("Accuracy: " + str(n.accuracy()))
'''

'''
t = np.linspace(0, 2 * np.pi, 20)
x = np.sin(t)
y = np.cos(t)
print(y)
plt.scatter(t,x,c=y)
plt.scatter([0],[0],c = [1])
plt.scatter([.25],[.25],c = [.5])
plt.scatter([.5],[.5],c = [-.5])
plt.scatter([1],[1], c = [-1])
plt.show()
'''