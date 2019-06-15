from matplotlib import pyplot as plt
import numpy as np


# goes length, width, color(red =1, blue -0)
data = [{'data': [3,1.5], 'prediction': 1}, {'data':[2,1], 'prediction': 0},
        {'data': [4,1.5], 'prediction': 1}, {'data': [3,1], 'prediction': 0},
        {'data': [3.5,.5], 'prediction': 1}, {'data': [2,.5], 'prediction': 0},
        {'data': [5.5,1], 'prediction': 1}, {'data': [1,1], 'prediction': 0}]

mystery_flower = {'data': [4.5,1]}


LEARNING_RATE = 0.2


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_p(x):#derivative of sigmoid
    return sigmoid(x)*(1-sigmoid(x))

### Testing results with Test Data
def accuracy():
    correct = 0;
    trials = 100
    for i in range(trials):
        ri = np.random.randint(len(data)) #random index that falls inside data's index's range
        point = data[ri]

        z = point[0]*w1 + point[1]*w2 +b
        pred = sigmoid(z)
        target = point[2]

        if (pred > .5 and target == 1) or (pred < .5 and target == 0): 
            correct += 1
    #print("Correct "+ str(correct*100/trials))
    return correct

def propogate(p, w):
    ### Forward propogation
    if(len(p['data']) == len(w)-1):
        pred = 0
        for i in range(len(p['data'])):
            pred += p[i]*w[i]
        pred = sigmoid(pred + w[-1]) # Add in the bias and normalize.
        out = {'prediction': pred}
    else:
        return False
    
    
    ### Backward propogation
    if()


###training loop

weight = [np.random.randn(), np.random.randn(), np.random.randn()]

#print(accuracy())
#accuracyList.append(accuracy())

for i in range(1000):
    ri = np.random.randint(len(data)) #random index that falls inside data's index's range
    result = propogate(data[ri], weight)    ### Need cost to be a list with lenght 4 (?)
    

    pred = sigmoid(z)
    target = point[2]
    cost = np.square(pred - target)
    
    dCost_pred = 2*(pred-target)
    dpred_z = sigmoid_p(z)
    
    dz_dw1 = point[0]
    dz_dw2 = point[1]
    dz_db = 1
    
    dCost_dw1 = dCost_pred*dpred_z*dz_dw1
    dCost_dw2 = dCost_pred*dpred_z*dz_dw2
    dCost_db = dCost_pred*dpred_z*dz_db
    
    w1 -= learning_rate*dCost_dw1
    w2 -= learning_rate*dCost_dw2
    b -= learning_rate*dCost_db
        
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
print(str(w1))
print(str(w2))
print(str(b))
accuracy()

point = mystery_flower
    
z = point[0]*w1 + point[1]*w2 +b
pred = sigmoid(z)

print("Prediction for Mystery Flower: "+ str(pred))

#print(accuracy())

#accuracyList.append(accuracy())

#plt.plot(accuracyList)