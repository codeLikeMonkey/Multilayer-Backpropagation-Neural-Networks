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


def data_prep():
    images, labels = read_in_data()
    images, labels = normalize_data(images, labels)

    return images,labels