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


def cost(p,w):
    pred = fwdProp(p, w)
    if(pred != False):
        targetValue = p['target']
        return np.square(targetValue - pred)
    else:
        return False
  
def backProp(p,w):
    ### Backward propogation
    H = 1e-15 # Might have to change later.
  

    if(p['target'] == 1 or p['target'] == 0 ):
        d_costs = []
        for i in range(len(w)):
            w_temp1 = w.copy()
            w_temp2 = w.copy()
            #if(sameArray(w_temp1,w)):
             #   print("Weights and Temp 1 are the same")
            #printArray(w)
            #printArray(w_temp1)
            #printArray(w_temp2)
            #print("Temp1 before"+ str(w_temp1[i]),end ="  ")
           
            w_temp1[i] += H
            w_temp2[i] -= H
           # print("Temp1 after"+ str(w_temp1[i]),end ="  ")
            
            #if(sameArray(w_temp1,w)):
             #   print("After Adding H, Weights and Temp 1 are the same")
            #printArray(w_temp1)
            #printArray(w_temp2)
            
            d_costs.append((cost(p,w_temp1)-cost(p,w_temp2))/(2*H))
        print(d_costs)

        return d_costs     
    else:
        return False



        
def trainNetwork():
    weight = [np.random.randn(), np.random.randn(), np.random.randn()]
    print("Weights Before Training")
    print(weight)
        
    
    for i in range(1000):
        ri = np.random.randint(len(data)) #random index that falls inside data's index's range
        #prediction = fwdProp(data[ri], weight)    ### Need cost to be a list with lenght 4 (?)
        
        d_costs = backProp(data[ri], weight)
       
        if(d_costs):
            weight[0] -= LEARNING_RATE*d_costs[0]
            weight[1] -= LEARNING_RATE*d_costs[1]
            weight[2] -= LEARNING_RATE*d_costs[2]
    print("Weights After Training")
    print(weight)
    
    return weight

        
    

pred = fwdProp(mystery_flower, trainNetwork())

print("\nPrediction for Mystery Flower: "+ str(pred))

print(accuracy(trainNetwork()))

