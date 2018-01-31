from neural_network import *
from data_prep import *
import matplotlib.pyplot as plt



if __name__ == "__main__":

    net = Network(
        layers = [784,64,10],
        max_epoch = 20,
        eta = 0.01,
        func = 'sigmoid',
        is_sqrt_initialize = False,
        is_shuffle = True,
        momentum_type = None
    )
    training_images,training_labels,test_images,test_labels = data_prep()
    net.fit(training_images,training_labels,test_images,test_labels)


