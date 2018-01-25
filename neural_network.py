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

    def __init__(self,layers):
        self.layers = layers # without bias
        self.weights = [np.random.randn(y,x) for (x,y) in zip(self.layers[0:-1],self.layers[1:])]
        self.z = [np.random.randn(x,1) for x in self.layers]
        self.a = [np.random.randn(x,1) for x in self.layers]
        self.deltas = [np.zeros((x,1)) for x in self.layers]
        self.bias = [np.random.randn(x,1) for x in self.layers[1:]]

    def feed_forward(self):
        for L in range(len(self.layers)-1):
            weighted_activations = np.dot(self.weights[L],self.a[L]) + self.bias[L]
            self.z[L+1] = weighted_activations
            self.a[L + 1] = sigmoid(self.z[L + 1])
            # if ((L+1) != len(self.layers) - 1):
            #     self.a[L+1] = sigmoid(self.z[L+1])
            # else:
            #     self.a[L+1] = softmax(self.z[L+1])



    def back_probagate(self):
        eta = 0.3

        for L in range(len(self.layers) - 1,0,-1):
            self.deltas[L - 1] = np.dot(self.weights[L-1 ].T,self.deltas[L]) * sigmoid_prime(self.z[L-1])
            # self.deltas[L - 1] = np.multiply(np.dot(self.weights[L - 1].T, self.deltas[L]),self.a[L-1]*(1- self.a[L-1]))
            self.weights[L-1] = self.weights[L-1] - eta * np.dot(self.deltas[L],self.a[L-1].T)
            # print("grad:%s"%np.dot(self.deltas[L],self.a[L-1].T))
            self.bias[ L-1 ] = self.bias[L-1 ] - eta * self.deltas[L]
        # print(self.weights)

    def test(self,input_data):
        self.a[0] = np.array(input_data).reshape(self.a[0].shape)
        self.feed_forward()
        # return softmaxself.z[-1].astype(float))
        return self.a[-1].astype(float)

    def train(self,images,labels):
        # for k in range(10):
        # for i in range(len(data_set)):
        #     self.deltas[-1] = np.multiply(self.test(data_set[i]) - target_set[i],self.g_prime(self.z[-1]))
        #     self.back_probagate()
        for i in range(images.shape[1]):
            # print(self.test(images[:,1]))
            input_data = images[:,i]
            target = np.zeros(10)
            target[labels[i]] = 1
            target = target.reshape(10,1)
            output = self.test(input_data)
            self.deltas[-1] = (self.a[-1] - target) * sigmoid_prime(self.z[-1])
            # self.deltas[-1] = np.multiply(output - target , self.g_prime(self.z[-1]))
            self.back_probagate()


if __name__ == "__main__":
    pass
    # net = Network([3,1])
    # net.weights[0][:,0] = np.array([1,0,-1]).reshape(3,)
    # net.feed_forward()
    # data_set = np.random.random((100,3))
    # data_set =np.random.random((10000,3))
    # target_set = np.sum(data_set,axis = 1)
    # target_set = target_set *0.2
    # net.train(data_set,target_set)
    # test = np.random.random([1,3])
    # print(test)
    # print(net.test(test))
    # print(np.sum(test)*0.2)