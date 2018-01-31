import numpy as np
from mnist import MNIST

def read_in_data(s = "training"):
    mndata = MNIST('./mnist_data')
    mndata.gz = True
    if s == "training":
        images, labels = mndata.load_training()
        
        return np.array(images).T, np.array(labels)
    if s == "test":
        images, labels = mndata.load_testing()
        
        return np.array(images[0:20000]).T, np.array(labels[0:20000])


def normalize_data(images,labels):

    new_images = images / 127.5 -1

    new_labels = np.zeros((10,len(labels)))
    def f(x):
        new_labels[labels[x], x] = 1
    list(map(f,[ i for i in range(len(labels))]))

    return new_images,new_labels


def data_prep():
    training_images, training_labels = read_in_data(s = "training")
    training_images, training_labels = normalize_data(training_images, training_labels)
    test_images,test_labels = read_in_data(s = "test")
    test_images,test_labels = normalize_data(test_images,test_labels)

    return training_images,training_labels,test_images,test_labels

if __name__ == "__main__":
    data_prep()