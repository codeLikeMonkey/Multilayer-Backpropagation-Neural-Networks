from neural_network import *
from data_prep import *
import matplotlib.pyplot as plt



if __name__ == "__main__":

    net = Network(
        layers = [784,64,10],
        max_epoch = 20,
        eta = 0.001,
        func = 'tanh2',
        is_sqrt_initialize = False,
        is_shuffle = True,
        momentum_type = "momentum"
    )
    training_images,training_labels,test_images,test_labels = data_prep()
    net.fit(training_images,training_labels,test_images,test_labels)


