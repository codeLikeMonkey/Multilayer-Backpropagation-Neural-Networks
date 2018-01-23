import numpy as np



def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def softmax():
    pass



class Network:

    def __init__(self,layers):
        self.layers = layers # without bias
        self.weights = [np.random.random((x+1,y+1)) for (x,y) in zip(self.layers[0:-1],self.layers[1:])]
        self.activations = [np.random.random([x+1,1]) for x in self.layers]
        self.deltas = [np.zeros([x+1,1]) for x in self.layers]
        self.g = sigmoid

    def feed_forward(self):
        for L in range(len(self.layers)-1):
            self.activations[L+1] = self.g(np.dot(self.weights[L].T,self.activations[L]))




    def back_probagate(self):
        pass

    def train(self):
        pass



if __name__ == "__main__":
    net = Network([3,1])
    net.activations[0]=np.array([1,2,3,1]).reshape(4,1)
    net.weights[0][:,0] = np.array([1,0,-1,1]).reshape(4,)
    net.weights[0][:,1] = np.array([1,2,3,4]).reshape(4,)
    net.feed_forward()