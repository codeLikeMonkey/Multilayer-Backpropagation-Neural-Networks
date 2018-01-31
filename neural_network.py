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

    def __init__(self,layers = [784,10],max_epoch = 5,eta = 0.01,func = "sigmoid",is_sqrt_initialize = False,is_shuffle = True,momentum_type = None):
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

        self.v_delta_weights = [np.zeros((y,x)) for (x,y) in zip(self.layers[0:-1],self.layers[1:])]
        self.v_bias = [np.zeros((x, 1)) for x in self.layers[1:]]
        self.momentum_type = momentum_type

        if func == "sigmoid":
            self.g = sigmoid
            self.g_prime = sigmoid_prime

        if func == "tanh":
            self.g = tan_sigmoid
            self.g_prime = tan_sigmoid_prime

        self.records = {
            "config": {
                "layers": layers,
                "eta": eta,
                "func":func,
                "is_sqrt_initialize":is_sqrt_initialize,
                "is_shuffle" : is_shuffle,
                "momentum_type" : momentum_type
            },
            "accuracy": {
                "training": [],
                "test": [],
                "hold_out": []
            },
            "loss": {
                "training": [],
                "test": [],
                "hold_out": []
            }

        }


        self.data = {
            "training_images":None,
            "training_labels":None,
            "hold_out_images":None,
            "hold_out_labels": None,
            "test_images":None,
            "test_labels":None
        }



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
        counter = 0
        for (mini_batch_data,mini_batch_labels) in  training_data:
            #train each data in the mini_batch

            if counter % 30 ==0:
                self.show_result()

            counter +=1

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

                if self.momentum_type == "momentum":

                    self.v_delta_weights[L-1] = 0.9 * self.v_delta_weights[L-1] - self.eta * Delta_batch_weights[L-1]
                    self.v_bias[L-1] = 0.9 * self.v_bias[L-1] - self.eta * Delta_batch_bias[L-1]

                if self.momentum_type == None:

                    self.v_delta_weights[L - 1] = - self.eta * Delta_batch_weights[L - 1]
                    self.v_bias[L - 1] = - self.eta * Delta_batch_bias[L - 1]


                self.weights[L-1] +=  self.v_delta_weights[L-1]
                self.bias[L-1] += self.v_bias[L-1]


    def fit(self,raw_images,raw_labels,test_images,test_labels):
        index = np.arange(raw_images.shape[1])
        for i in range(self.max_epoch):
            print("\nEpoch[%d]"%i)

            if self.is_shuffle:
                np.random.shuffle(index)

            images = raw_images[:, index]
            labels = raw_labels[:, index]

            hold_out_images = images[:, 0:10000]
            hold_out_labels = labels[:, 0:10000]

            training_images = images[:, 10000:]
            training_labels = labels[:, 10000:]


            self.data["training_images"] = training_images.copy()
            self.data["training_labels"] = training_labels.copy()
            self.data["test_images"] = test_images.copy()
            self.data["test_labels"] = test_labels.copy()
            self.data["hold_out_images"] = hold_out_images.copy()
            self.data["hold_out_labels"] = hold_out_labels.copy()


            self.train_with_mini_batch(training_images, training_labels)

            del training_images
            del training_labels
            del hold_out_labels
            del hold_out_images





    def calculate_loss(self,result,labels):
        Y = result.T
        Target = labels

        E = -np.sum(np.sum(np.multiply(Target, np.log10(Y))))

        return E.astype(float)#/Target.shape[1]

    def check(self,images,labels):
        raw_result = [self.test(images[:,i]) for i in range(labels.shape[1])]
        result = np.array([np.argmax(x) for x in raw_result])
        # result = np.array([np.argmax(self.test(images[:,i])) for i in range(labels.shape[1])])
        y = np.array([x.reshape(10, ) for x in raw_result])

        #calculate cross entropy
        E = self.calculate_loss(y,labels)
        # print(E)
        return np.sum(result == np.argmax(labels, axis = 0)) / labels.shape[1],E


    def show_result(self):

        training_images = self.data["training_images"]
        training_labels = self.data["training_labels"]
        test_images = self.data["test_images"]
        test_labels = self.data["test_labels"]
        hold_out_images = self.data["hold_out_images"]
        hold_out_labels = self.data["hold_out_labels"]
        training_accuracy, training_loss = self.check(training_images, training_labels)
        test_accuracy, test_loss = self.check(test_images, test_labels)
        hold_out_accuracy, hold_out_loss = self.check(hold_out_images, hold_out_labels)
        print("Accuracy : \tTraining : %10.4f\t Test : %10.4f\t Hold out : %10.4f" % (
            training_accuracy * 100, test_accuracy * 100, hold_out_accuracy * 100))
        print("Loss     : \tTraining : %10.4f\t Test : %10.4f\t Hold out : %10.4f" % (
            training_loss, test_loss, hold_out_loss))

        # record accuracy
        self.records["accuracy"]["training"].append(training_accuracy * 100)
        self.records["accuracy"]["test"].append(test_accuracy * 100)
        self.records["accuracy"]["hold_out"].append(hold_out_accuracy * 100)

        # record loss

        self.records["loss"]["training"].append(training_loss)
        self.records["loss"]["test"].append(test_loss)
        self.records["loss"]["hold_out"].append(hold_out_loss)







if __name__ == "__main__":
    pass