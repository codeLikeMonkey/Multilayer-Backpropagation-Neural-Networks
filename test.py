from neural_network import *
import numpy as np
from mnist import MNIST

def read_in_data():
    mndata = MNIST('./mnist_data')
    mndata.gz = True
    images, labels = mndata.load_training()
    return np.array(images[0:60000]).T, np.array(labels[0:60000])

def normalize_data(images,labels):

    new_images = images / 127.5 -1

    new_labels = np.zeros((10,len(labels)))
    def f(x):
        new_labels[labels[x], x] = 1
    list(map(f,[ i for i in range(len(labels))]))

    return new_images,new_labels


if __name__ == "__main__":
    images, labels = read_in_data()
    #check error
    net = Network(
        layers = [784,50,50,10],
        max_epoch = 200,
        eta = 0.01,
        func = 'sigmoid',
        is_sqrt_initialize = True,
        is_shuffle = True,
        is_momentum = True
    )
    images, labels = normalize_data(images, labels)
    net.fit(images,labels)
    # map(net.check_accuracy(images,labels)


