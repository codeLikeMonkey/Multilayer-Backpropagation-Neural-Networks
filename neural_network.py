import numpy as np



def sigmoid(k):
    return 1.0/(1.0 + np.exp(-k))

def sigmoid_prime(k):
    return sigmoid(k) * (1 - sigmoid(k))

def tan_sigmoid(k):
    return (np.exp(k) - np.exp(-k))/(np.exp(k) + np.exp(-k))

def tan_sigmoid_prime(k):
    return 1 - tan_sigmoid(k)**2

def softmax(last_layers):
    alpha = last_layers - np.max(last_layers)
    alpha = np.exp(alpha)
    beta = alpha/np.sum(alpha)

    return beta

class Network:
    np.random.seed(0)

    def __init__(self,layers = [784,10],max_epoch = 5,eta = 0.01,func = "sigmoid",is_sqrt_initialize = False,is_shuffle = True,is_momentum = False):
        self.layers = layers # without bias

        if is_sqrt_initialize:
            self.weights = [np.random.normal(0, 1.0 / np.sqrt(y), (y, x)) for (x, y) in zip(self.layers[0:-1], self.layers[1:])]
        else:
            self.weights = [np.random.randn(y,x) for (x,y) in zip(self.layers[0:-1],self.layers[1:])]

        self.z = [np.random.randn(x,1) for x in self.layers]
        self.a = [np.random.randn(x,1) for x in self.layers]
        self.bias = [np.random.randn(x,1) for x in self.layers[1:]]
        self.deltas = [np.zeros((x, 1)) for x in self.layers]
        self.mini_batch_size = 128
        self.eta = eta
        self.max_epoch = max_epoch
        self.is_shuffle = is_shuffle

        if func == "sigmoid":
            self.g = sigmoid
            self.g_prime = sigmoid_prime

        if func == "tanh":
            self.g = tan_sigmoid
            self.g_prime = tan_sigmoid_prime



    def feed_forward(self):
        for L in range(len(self.layers)-1):
            weighted_activations = np.dot(self.weights[L],self.a[L]) + self.bias[L]
            self.z[L+1] = weighted_activations
            # self.a[L + 1] = sigmoid(self.z[L + 1])
            if ((L+1) != len(self.layers) - 1):
                self.a[L+1] = self.g(self.z[L+1])
            else:
                self.a[L+1] = softmax(self.z[L+1])

    def mini_batch_back_probagate(self):
        delta_weights = [np.zeros((y,x)) for (x,y) in zip(self.layers[0:-1],self.layers[1:])]
        delta_bias = [np.zeros((x,1)) for x in self.layers[1:]]
        for L in range(len(self.layers) - 1,0,-1):
            self.deltas[L - 1] = np.dot(self.weights[L-1 ].T,self.deltas[L]) * self.g_prime(self.z[L-1])
            delta_weights[L-1] = np.dot(self.deltas[L],self.a[L-1].T)
            delta_bias[L-1] = self.deltas[L].copy()

        return delta_weights,delta_bias


    def test(self,input_data):
        self.a[0] = np.array(input_data).reshape(self.a[0].shape)
        self.feed_forward()
        return self.a[-1].astype(float)

    def train_with_mini_batch(self,raw_images,raw_labels):

        images = raw_images.copy()
        labels = raw_labels.copy()
        index = np.arange(images.shape[1]//self.mini_batch_size * self.mini_batch_size)
        # divide the data in to small batches
        training_data = [(images[:,index_batch],labels[:,index_batch]) for index_batch in np.split(index,len(index)//self.mini_batch_size)]
        #one epoch
        for (mini_batch_data,mini_batch_labels) in  training_data:
            #train each data in the mini_batch

            Delta_batch_weights = [np.zeros((y,x)) for (x,y) in zip(self.layers[0:-1],self.layers[1:])]
            Delta_batch_bias =  [np.zeros((x,1)) for x in self.layers[1:]]
            for i in range(self.mini_batch_size):
                input_data = mini_batch_data[:,i]
                output_target = mini_batch_labels[:,i].reshape(10,1)
                self.deltas[-1] = (self.test(input_data) - output_target) #* sigmoid_prime(self.z[-1])
                delta_weights, delta_bias = self.mini_batch_back_probagate()

                for L in range(len(self.layers) - 1,0,-1):
                    Delta_batch_weights[L-1] += delta_weights[L-1]
                    Delta_batch_bias[L-1] += delta_bias[L-1]


            #update weights and bias
            for L in range(len(self.layers) - 1,0,-1):
                self.weights[L-1] = self.weights[L-1] - self.eta * Delta_batch_weights[L-1]
                self.bias[ L-1 ] = self.bias[L-1 ] - self.eta * Delta_batch_bias[L-1]

    def fit(self,raw_images,raw_labels):
        index = np.arange(raw_images.shape[1])
        for i in range(self.max_epoch):

            if self.is_shuffle:
                np.random.shuffle(index)

            images = raw_images[:, index]
            labels = raw_labels[:, index]

            hold_out_images = images[:, 0:5000]
            hold_out_labels = labels[:, 0:5000]

            training_images = images[:, 5000:50000]
            training_labels = labels[:, 5000:50000]

            validation_images = images[:, 50000:]
            validation_labels = labels[:, 50000:]

            self.train_with_mini_batch(training_images, training_labels)
            training_accuracy = self.check_accuracy(training_images,training_labels)
            validation_accuracy = self.check_accuracy(validation_images, validation_labels)
            hold_out_accuracy = self.check_accuracy(hold_out_images, hold_out_labels)
            print("Epoch[%d]\tTraining : %.4f\t Val : %.4f\t Hold out : %.4f" % (i,training_accuracy * 100,validation_accuracy * 100,hold_out_accuracy * 100))


    def check_accuracy(self,images,lables):
        result = np.array([np.argmax(self.test(images[:,i])) for i in range(lables.shape[1])])
        return np.sum(result == np.argmax(lables, axis = 0)) / lables.shape[1]

    def calculate_loss(self,target):
        pass




if __name__ == "__main__":
    pass