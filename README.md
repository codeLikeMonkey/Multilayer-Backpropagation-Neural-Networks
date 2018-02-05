# Multilayer-Backpropagation-Neural-Networks
CSE253 Homework 2

This project is the second project of UCSD CSE 253 Neural Network and pattern Recognition.

We implemented a neural network class using Numpy. The class includes common tricks for neural network, including batch gradient descent, cross validation,
momentum, sqrt weight initialization, different activation function (tanh,sigmoid,softmax )

To run the program, you need to import Network class from neural_network.py in your code. 
You can specialize your parameters like :

from neural_network import *
from data_prep import *
import matplotlib.pyplot as plt


if __name__ == "__main__":

    net = Network(
        #choose your layer parameters 
        #[784,10] for pure softmax
        #ex:[784,64,64] There are 2 hidden layers of 64 units between input layer and outlayer. 
        #you can include any number of hidden layers if you don't care about speed. 
        layers = [784,1000,10],
        
        #choose your max epochs
        max_epoch = 30,
       
        #learning rate
        eta = 0.001,
        #choose your activation function inside hidden layers, you may choose 'sigmoid','tanh','tanh2'
        func = 'sigmoid',
        
        #initialization 
        is_sqrt_initialize = True,
        
        # shuffle the data
        is_shuffle = True,
        
        #Now we only have traditional momentum type
        momentum_type = "momentum"
    )
    
    Then you can use class methods to show the result like :
    
    training_images,training_labels,test_images,test_labels = data_prep()
    net.fit(training_images,training_labels,test_images,test_labels)
    
    
