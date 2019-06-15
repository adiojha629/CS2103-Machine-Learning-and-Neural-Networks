from matplotlib import pyplot as plt
import numpy as np


# goes length, width, color(red =1, blue -0)
data = [{'data': [3,1.5], 'target': 1}, {'data':[2,1], 'target': 0},
        {'data': [4,1.5], 'target': 1}, {'data': [3,1], 'target': 0},
        {'data': [3.5,.5], 'target': 1}, {'data': [2,.5], 'target': 0},
        {'data': [5.5,1], 'target': 1}, {'data': [1,1], 'target': 0}]

mystery_flower = {'data': [4.5,1]}


LEARNING_RATE = 0.2


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_p(x):#derivative of sigmoid
    return sigmoid(x)*(1-sigmoid(x))

### Testing results with Test Data
def accuracy(w):
    correct = 0;
    trials = 100
    for i in range(trials):
        ri = np.random.randint(len(data)) #random index that falls inside data's index's range
        point = data[ri]

        pred = fwdProp(point, w)

        target = point['target']

        if (pred > .5 and target == 1) or (pred < .5 and target == 0): 
            correct += 1
    #print("Correct "+ str(correct*100/trials))
    return correct

    
def fwdProp(p, w):
    ### Forward propogation
    if(len(p['data']) == len(w)-1):
        pred = 0
        for i in range(len(p['data'])):
            pred += p['data'][i]*w[i]
        pred = sigmoid(pred + w[-1]) # Add in the bias and normalize.
        return pred
    else:
        return False
  
def backProp(p,w):
    ### Backward propogation
    H = 1e-18 # Might have to change later.
    if(p['target']):
        d_costs = []
        for i in range(len(w)):
            w_temp1 = w
            w_temp2 = w
            w_temp1[i] += H
            w_temp2[i] -= H
            d_costs.append((fwdProp(p,w_temp1)-fwdProp(p,w_temp2))/(2*H))
        return d_costs     
    else:
        return False


###training loop

weight = [np.random.randn(), np.random.randn(), np.random.randn()]

#print(accuracy())
#accuracyList.append(accuracy())

for i in range(10000):
    ri = np.random.randint(len(data)) #random index that falls inside data's index's range
    prediction = fwdProp(data[ri], weight)    ### Need cost to be a list with lenght 4 (?)
    
    d_costs = backProp(data[ri], weight)
    
    if(d_costs):
        weight[0] -= LEARNING_RATE*d_costs[0]
        weight[1] -= LEARNING_RATE*d_costs[1]
        weight[2] -= LEARNING_RATE*d_costs[2]
        
    #if i % 100 == 0:
        #print("Training Session: "+ str(i))
        #accuracyList.append(accuracy())

        #costSum = 0
       #for a in range(len(data)):
        #    point = data[a]
         #   z = point[0]*w1 + point[1]*w2 +b
          #  pred = sigmoid(z)
           # target = point[2]
            #costSum += np.square(pred - target)
        #costsList.append(costSum/len(data))   
             
print("End of Training")
print(str(weight[0]))
print(str(weight[1]))
print(str(weight[2]))
accuracy(weight)

pred = fwdProp(mystery_flower, weight)

print("Prediction for Mystery Flower: "+ str(pred))

#print(accuracy())

#accuracyList.append(accuracy())

#plt.plot(accuracyList)