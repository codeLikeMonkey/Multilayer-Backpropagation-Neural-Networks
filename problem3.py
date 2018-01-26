from neural_network import *
import numpy as np
from mnist import MNIST

def read_in_data():
    mndata = MNIST('./mnist_data')
    mndata.gz = True
    images, labels = mndata.load_training()
    return np.array(images[0:5]).T, np.array(labels[0:5])

def normalize_data(images,labels):

    new_images = images / 127.5 - 1

    return new_images,labels


if __name__ == "__main__":
    images, labels = read_in_data()

    #check error
    net = Network([784,64,10])
    images, labels = normalize_data(images, labels)

    epsilon = 0.1
    #calculate by expression

    #calculate the first one
    input_data = images[:, 0]
    target = np.zeros(10)
    target[labels[0]] = 1
    target = target.reshape(10, 1)
    net.weights[0][1,2] +=epsilon
    E1 = net.test(input_data)
    net.weights[0][1,2] -= 2*epsilon
    E2 = net.test(input_data)
    print(E1)
    print(E2)
    print((E1-E2))
















