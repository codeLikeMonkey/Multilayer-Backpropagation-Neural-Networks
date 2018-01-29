from neural_network import *
from data_prep import *



if __name__ == "__main__":

    net = Network(
        layers = [784,10],
        max_epoch = 200,
        eta = 0.0001,
        func = 'sigmoid',
        is_sqrt_initialize = True,
        is_shuffle = True,
        momentum_type = 'momentum'
    )
    images,labels = data_prep()
    net.fit(images,labels)


