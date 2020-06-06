# coding: utf-8
import numpy as np
import time as t
from tqdm import tqdm

#Define Function
def sigmoid(x):
    return 1/(np.exp(x*-1)+1)

def One_Hot_Encoding(L):
    global unique,unique_list,temp,output_list
    unique=0
    unique_list=[]
    for i in L:
        if i not in unique_list:
            unique+=1;
            unique_list.append(i)
        else:
            continue
    output_list = np.full(unique,0)
    for i in range(len(L)):
        temp=np.full(unique,0)
        temp[unique_list.index(L[i])]=1
        output_list=np.vstack((output_list,temp))
    output_list=output_list[1::]
    return output_list

def Load_data():
    global X,y
#    INPUT YOUR TXT DATA HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    X = np.loadtxt("YOUR TXT DATA",delimiter='\t',dtype='str')
    y=np.squeeze(np.array(X[::,-1::],dtype=str))
    # Add 'one col' in X    reason: (calculate bias)
    X=X[::,0:-1:]
    X = X.astype(np.float)
    y=One_Hot_Encoding(y)
    

#Initial
def init():
    Load_data()
    global activation_func,input_dim,output_dim,input_bias,hidden_dim,hidden_bias,LR,w1,w2,X,R
    
    input_dim=X[0].shape[0]
    hidden_dim=3
    output_dim=unique
    activation_func=sigmoid
    input_bias=False
    hidden_bias=False
    k = 1#a number that multiply to weight matrix
    LR =.8 #Learning Rate
    R=0.8  #Train Test Ratio
    shuffle(0.7)
        
    w1=np.matrix(np.random.rand(hidden_dim,input_dim+input_bias)*k)
    w2=np.matrix(np.random.rand(output_dim,hidden_dim+hidden_bias)*k)
    One=np.full((X.shape[0],1),1)
    if input_bias:
        X=np.hstack((One,X))
        X = X.astype(np.float)
        del One
        

def shuffle(num):
    global train_index,val_index
    index=np.arange(len(X))   
    np.random.shuffle(index)
    train_index=index[0:int(len(X)*num)]
    val_index=index[int(len(X)*num):len(X)]

init()

#Forward
start_time=t.time()
for T in tqdm(range(10)):#Epoch times
    for I in train_index:
#         print(I)
        a1=activation_func(w1.dot(X[I]))
        if hidden_bias:
            One=np.full((a1.shape[0],1),1)
            a1=np.hstack((One,a1))
            del One
        a1=a1.T
        a2=activation_func(w2.dot(a1))
        
#Backward        
        # Calculate Error
        delta=[]
        for i in range(len(a2)):
           delta.append(float((y[I,i]-a2[i])*a2[i]*(1-a2[i])))
        delta_hid=[]
        for i in range(w2.shape[1]):
            delta_hid.append(float(np.dot(delta,w2[:,i])*a1[i]*(1-a1[i])))

        # Update Weight
        
        # W2
        for i in range(w2.shape[1]):
            for j in range(w2.shape[0]):
                w2[j,i] = w2[j,i]+(LR*delta[j]*a1[i])
        # W1
        for i in range(w1.shape[1]):
            for j in range(w1.shape[0]):
                w1[j,i] = w1[j,i]+(LR*delta_hid[j]*X[I,i])
Using_Time=t.time()-start_time