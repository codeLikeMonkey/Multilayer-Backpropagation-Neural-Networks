import numpy as np



def sigmoid(k):
    return 1.0/(1.0 + np.exp(-k))

def sigmoid_prime(k):
    return sigmoid(k) * (1 - sigmoid(k))

def softmax(last_layers):
    alpha = last_layers - np.max(last_layers)
    alpha = np.exp(alpha)
    beta = alpha/np.sum(alpha)

    return beta



class Network:
    np.random.seed(0)

    def __init__(self,layers,max_epoch,eta,func):
        self.layers = layers # without bias
        self.weights = [np.random.randn(y,x) for (x,y) in zip(self.layers[0:-1],self.layers[1:])]
        self.z = [np.random.randn(x,1) for x in self.layers]
        self.a = [np.random.randn(x,1) for x in self.layers]
        self.deltas = [np.zeros((x,1)) for x in self.layers]
        self.bias = [np.random.randn(x,1) for x in self.layers[1:]]
        self.mini_batch_size = 128
        self.eta = eta
        self.max_epoch = max_epoch

        if func == "sigmoid":
            self.g = sigmoid
            self.g_prime = sigmoid_prime


    def feed_forward(self):
        for L in range(len(self.layers)-1):
            weighted_activations = np.dot(self.weights[L],self.a[L]) + self.bias[L]
            self.z[L+1] = weighted_activations
            # self.a[L + 1] = sigmoid(self.z[L + 1])
            if ((L+1) != len(self.layers) - 1):
                self.a[L+1] = self.g(self.z[L+1])
            else:
                self.a[L+1] = softmax(self.z[L+1])

    # def back_probagate(self):
    #     eta = 0.001
    #     delta_weights = []
    #     delta_bias = []
    #
    #     for L in range(len(self.layers) - 1,0,-1):
    #         self.deltas[L - 1] = np.dot(self.weights[L-1 ].T,self.deltas[L]) * sigmoid_prime(self.z[L-1])
    #         self.weights[L-1] = self.weights[L-1] - eta * np.dot(self.deltas[L],self.a[L-1].T)
    #         self.bias[ L-1 ] = self.bias[L-1 ] - eta * self.deltas[L]


    def mini_batch_back_probagate(self):
        delta_weights = [np.zeros((y,x)) for (x,y) in zip(self.layers[0:-1],self.layers[1:])]
        delta_bias = [np.zeros((x,1)) for x in self.layers[1:]]
        for L in range(len(self.layers) - 1,0,-1):
            self.deltas[L - 1] = np.dot(self.weights[L-1 ].T,self.deltas[L]) * self.g_prime(self.z[L-1])
            # self.weights[L-1] = self.weights[L-1] - eta * np.dot(self.deltas[L],self.a[L-1].T)
            # self.bias[ L-1 ] = self.bias[L-1 ] - eta * self.deltas[L]
            delta_weights[L-1] = np.dot(self.deltas[L],self.a[L-1].T)
            delta_bias[L-1] = self.deltas[L].copy()

        return delta_weights,delta_bias


    def test(self,input_data):
        self.a[0] = np.array(input_data).reshape(self.a[0].shape)
        self.feed_forward()
        return self.a[-1].astype(float)

    def train_with_mini_batch(self,raw_images,raw_labels):

        #shuffle the data

        images = raw_images.copy()
        labels = raw_labels.copy()
        index = np.arange(images.shape[1]//self.mini_batch_size * self.mini_batch_size)
        # np.random.shuffle(index)
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
                # output_target = np.zeros(10)
                # output_target[mini_batch_labels[i]] = 1
                # output_target = output_target.reshape(10,1)
                self.deltas[-1] = (self.test(input_data) - output_target) #* sigmoid_prime(self.z[-1])
                delta_weights, delta_bias = self.mini_batch_back_probagate()

                for L in range(len(self.layers) - 1,0,-1):
                    Delta_batch_weights[L-1] +=delta_weights[L-1]
                    Delta_batch_bias[L-1] += delta_bias[L-1]


            #update weights and bias
            for L in range(len(self.layers) - 1,0,-1):
                self.weights[L-1] = self.weights[L-1] - self.eta * Delta_batch_weights[L-1]
                self.bias[ L-1 ] = self.bias[L-1 ] - self.eta * Delta_batch_bias[L-1]

    # def train(self,images,labels):
    #     for i in range(images.shape[1]):
    #         # print(self.test(images[:,1]))
    #         input_data = images[:,i]
    #         target = np.zeros(10)
    #         target[labels[i]] = 1
    #         target = target.reshape(10,1)
    #         output = self.test(input_data)
    #         self.deltas[-1] = (self.a[-1] - target) #* sigmoid_prime(self.z[-1])
    #         self.back_probagate()

    def fit(self,raw_images,raw_labels):
        # split into training and validation sets
        # images = raw_images.copy()
        # labels = raw_images.copy()
        index = np.arange(raw_images.shape[1])
        for i in range(self.max_epoch):

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
            print("Epoch[%d]\tTraining : %s\t Val : %s\t Hold out : %s" % (i,training_accuracy,validation_accuracy,hold_out_accuracy))


    def check_accuracy(self,images,lables):
        result = np.array([np.argmax(self.test(images[:,i])) for i in range(lables.shape[1])])
        return np.sum(result == np.argmax(lables, axis = 0)) / lables.shape[1]




if __name__ == "__main__":
    pass