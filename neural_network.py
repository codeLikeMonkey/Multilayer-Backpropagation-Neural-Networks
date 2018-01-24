import numpy as np



def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def softmax():
    pass



class Network:

    def __init__(self,layers):
        self.layers = layers # without bias
        self.weights = [np.random.random((x,y)) for (x,y) in zip(self.layers[0:-1],self.layers[1:])]
        self.z = [np.random.random([x,1]) for x in self.layers]
        self.a = [np.random.random([x,1]) for x in self.layers]
        self.deltas = [np.zeros([x,1]) for x in self.layers]
        self.bias = [np.random.random([x,1]) for x in self.layers[1:]]
        self.g = sigmoid
        self.g_prime = sigmoid_prime

    def feed_forward(self):
        for L in range(len(self.layers)-1):
            weighted_activations = np.dot(self.weights[L].T,self.a[L]) + self.bias[L]
            self.z[L+1] = weighted_activations
            self.a[L+1] = self.g(self.z[L+1])


    def back_probagate(self):
        eta = 0.5
        for L in range(len(self.layers) - 1,0,-1):
            self.deltas[L - 1] = np.multiply(np.dot(self.weights[L - 1],self.deltas[L]),self.g_prime(self.z[L - 1]))
            self.weights[L - 1] = self.weights[L - 1] - eta * np.dot(self.a[L-1],self.deltas[L].T)
            self.bias[L - 1] = self.bias[L - 1] - eta * self.deltas[L]
        print(self.weights)

    def test(self,input_data):
        self.a[0] = np.array(input_data).reshape(self.a[0].shape)
        self.feed_forward()
        return self.a[-1].astype(float)

    def train(self,data_set,target_set):
        for k in range(10):
            for i in range(len(data_set)):
                self.deltas[-1] = np.multiply(self.test(data_set[i]) - target_set[i],self.g_prime(self.z[-1]))
                self.back_probagate()






if __name__ == "__main__":
    net = Network([4,1])
    # net.weights[0][:,0] = np.array([1,0,-1]).reshape(3,)
    # net.feed_forward()
    # data_set = np.random.random((100,3))
    data_set =np.random.random((1000,4))
    target_set = np.sum(data_set,axis = 1)
    target_set = target_set/np.max(target_set)
    net.train(data_set,target_set)
    print(data_set[10,:])
    print(net.test(data_set[10,:]))
    print(target_set[10])