from neural_network import *
from data_prep import *
import matplotlib.pyplot as plt



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
    training_images,training_labels,test_images,test_labels = data_prep()
    net.fit(training_images,training_labels,test_images,test_labels)

    # plt.plot(np.arange(0,net.max_epoch),net.records["loss"]["training"])
    # plt.plot(np.arange(0, net.max_epoch), net.records["loss"]["val"])
    # plt.plot(np.arange(0, net.max_epoch), net.records["loss"]["hold_out"])
    # plt.show()


