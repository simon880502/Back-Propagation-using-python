# coding: utf-8
import numpy as np
import time as t
from tqdm import tqdm
import matplotlib.pyplot as plt


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
    X = np.loadtxt("iris_training_data.txt",delimiter='\t',dtype='str')
    y=np.squeeze(np.array(X[::,-1::],dtype=str))
    # Add 'one col' in X    reason: (calculate bias)
    X=X[::,0:-1:]
    X = X.astype(np.float)
    y=One_Hot_Encoding(y)

def Load_Testing_data():
    global Testing_X,Testing_y
    Testing_X = np.loadtxt("iris_testing_data.txt",delimiter='\t',dtype='str')
    Testing_y=np.squeeze(np.array(Testing_X[::,-1::],dtype=str))
    Testing_X=Testing_X[::,0:-1:]
    Testing_X = Testing_X.astype(np.float)
    output_list = np.full(unique,0)
    for i in range(len(Testing_y)):
        temp=np.full(unique,0)
        temp[unique_list.index(Testing_y[i])]=1
        output_list=np.vstack((output_list,temp))
    Testing_y=output_list[1::]
    if input_bias:
        One=np.full((Testing_X.shape[0],1),1)
        Testing_X=np.hstack((One,Testing_X))
        Testing_X = Testing_X.astype(np.float)
        del One

def shuffle(num):
    global train_index,val_index
    index=np.arange(len(X))   
    np.random.shuffle(index)
    train_index=index[0:int(len(X)*num)]
    val_index=index[int(len(X)*num):len(X)]

#Initial
def init():
    Load_data()
    global activation_func,input_dim,output_dim,input_bias,hidden_dim,hidden_bias,LR,w1,w2,X,R
    input_dim=X[0].shape[0]
    output_dim=unique
    
    
    
#V A R I A B L E   S E T U P==================
    
    hidden_dim=6
    activation_func=sigmoid
    input_bias=False
    hidden_bias=False
    k = 1#a number that multiply to weight matrix
    LR =0.05 #Learning Rate
    R=0.75 #Train Val Ratio
    
#V A R I A B L E   S E T U P==================  
    
    
    shuffle(R) 
    w1=np.matrix(np.random.rand(hidden_dim,input_dim+input_bias)*k)
    w2=np.matrix(np.random.rand(output_dim,hidden_dim+hidden_bias)*k)
    One=np.full((X.shape[0],1),1)
    if input_bias:
        X=np.hstack((One,X))
        X = X.astype(np.float)
        del One
        
def check(index):
    acc=0
    for I in index:
        a1=activation_func(w1.dot(X[I]))
        if hidden_bias:
            One=np.full((a1.shape[0],1),1)
            a1=np.hstack((One,a1))
            del One
        a1=a1.T
        a2=activation_func(w2.dot(a1))
    #     print('Data ID:',I,'\noutput:\n',a2,'\ny:',y[I],'\n')
    #     print(np.argmax(a2),np.argmax(y[I]))
        if np.argmax(a2)==np.argmax(y[I]):
            acc+=1
    return (acc/len(index))

#Writing the result to REC.txt
def result():
    a=['Number of hidden neurons : ', 'Train & Val Ratio : ',\
 'Learning rates : ', 'Hidden Bias : ', 'Input Bias : ', 'Train Acc : ',\
 'Val Acc : ', 'Using epochs : ', 'Using Time :']
    b=[hidden_dim+hidden_bias,R,LR,hidden_bias,input_bias,\
       round(check(train_index),3),round(check(val_index),3),sav_T+1,round(Using_Time,3)]
    fp = open('REC.txt','a')
    for a,b in zip(a,b):
        fp.write(format(a,'<28s')+str(b)+'\n')
        print(format(a,'<28s')+str(b))
    fp.close()



#Forward
init()
start_time=t.time()
save_w=[]
train_history=[[],[]]
val_history=[[],[]]
for T in tqdm(range(100)):#Epoch times
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
           delta.append(float((y[I,i]-a2[i])))
        delta_hid=[]
        for i in range(w2.shape[1]):
            delta_hid.append(float(np.dot(delta,w2[:,i])*a1[i]*(1-a1[i])))

        # Update Weight
        
        # W2
        w2+=(LR*np.matrix(delta).T*a1.T)
        # W1
        w1+=(LR*np.matrix(delta_hid[hidden_bias:]).T*X[I])
        
    #PLOT 
    train_history[0].append(T)
    train_history[1].append(check(train_index))
    val_history[0].append(T)
    val_history[1].append(check(val_index))
    V=check(val_index)    
#    Save the best performance weight to prevent overfitting
#    print(V)
    save_w.append([w1,w2])
    if V==1:
        break
w1,w2=save_w[(val_history[1].index(max(val_history[1])))-1]
sav_T=val_history[1].index(max(val_history[1]))


plt.xlabel('Epoch')
plt.ylabel('Accuracy')
l1=plt.plot(train_history[0],train_history[1],label='Train acc',linewidth=2.0)
l2=plt.plot(val_history[0],val_history[1],linewidth=2.0,linestyle='--',label='Val acc')
plt.ylim((0,1))
plt.legend()
plt.show()

Using_Time=t.time()-start_time
print('\n')
print('Train ACC : %.2f'%check(train_index))             
print('Val   ACC : %.2f\n'%check(val_index))
result()

Load_Testing_data() #Input Testing data




#Calculate Test acc
One=np.full((Testing_X.shape[0],1),1)
Test_acc=0

for I in range(len(Testing_X)):
    a1=activation_func(w1.dot(Testing_X[I]))
    if hidden_bias:
        One=np.full((a1.shape[0],1),1)
        a1=np.hstack((One,a1))
        del One
    a1=a1.T
    a2=activation_func(w2.dot(a1))
#     print('Data ID:',I,'\noutput:\n',a2,'\ny:',y[I],'\n')
#     print(np.argmax(a2),np.argmax(y[I]))
    if np.argmax(a2)==np.argmax(Testing_y[I]):
        Test_acc+=1
Test_acc/=len(Testing_X)

fp = open('REC.txt','a')
fp.write(format("Testing Acc :",'<28s')+str(round(Test_acc,2))+'\n')
print(format("Testing Acc :",'<28s')+str(round(Test_acc,2)))
fp.write('*'*35+'\n')
fp.close()
print('\nRecord is written to the REC.txt')
