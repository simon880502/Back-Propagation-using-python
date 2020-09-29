# Back-Propagation-using-python-WITHOUT-sklearn
The function of this code is to build a neural network that has one hidden layer.

The function of this code is to build a neural network that has one hidden layer. 
You can choose the hidden neurals freely. But only has one hidden layer.
The package that use in this file is "numpy"、"tqdm"、"time"、"matplotlib.pyplot ".
Recommand python version :3.6.x

The data is not necessary to be 4 atrribute, can be less or more. But data must be seperated by 'tab'.
Target variable has to be discrete and put it in the last column.

"numpy"：Doing calculation
"tqdm" ：Showing the training
"time" ：Ｍeasuring the training time
"matplotlib.pyplot ":Plotting the graph

The parameter that you can set：


	１．activation_func=sigmoid
		You can define the activation function by yourself and replace it.
		
		
	２．hidden_dim=3
		the number of the hidden neurals 
		
		
	３．input_bias=False
		True if you want bias if First layer 
		
		
	４．hidden_bias=False
		True if you want bias if Second layer 
		
		
	５．k = 1
		#a number that multiply to the all element of the weight matrix
		
		
    	６．LR =.8 
		#Learning Rate
		
		
    	７．R=0.8  
		#Percentage of the training data   (1-R will be the percentage of the validation data)
