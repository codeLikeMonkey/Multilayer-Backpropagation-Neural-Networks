from neural_network import *
import numpy as np
from mnist import MNIST

def read_in_data():
    mndata = MNIST('./mnist_data')
    mndata.gz = True
    images, labels = mndata.load_training()
    return np.array(images[0:20000]).T, np.array(labels[0:20000])

def normalize_data(images,labels):

    new_images = images / 127.5 -1

    return new_images,labels


if __name__ == "__main__":
    images, labels = read_in_data()

    #check error
    net = Network(layers = [784,64,10],max_epoch = 30,eta = 0.01)
    images, labels = normalize_data(images, labels)
    # for i in range(100):
    #     net.train_with_mini_batch(images,labels)
    #     result = []
    #     for j in range(10):
    #         result.append(np.argmax((net.test(images[:,j]))))
    #     print(result)
    net.fit(images,labels)


