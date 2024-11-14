import numpy as np
from collections import OrderedDict
from layers import Relu, Affine, SoftmaxWithLoss
import pickle

class TwoLayerNetWithBackProp:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.update_layers()
        self.last_layer = SoftmaxWithLoss()

    def update_layers(self):
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['w1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['w2'], self.params['b2'])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['w1'] = np.zeros_like(self.params['w1'])
        grads['b1'] = np.zeros_like(self.params['b1'])
        grads['w2'] = np.zeros_like(self.params['w2'])
        grads['b2'] = np.zeros_like(self.params['b2'])

        for key in ('w1', 'b1', 'w2', 'b2'):
            h = 1e-4
            for i in range(self.params[key].size):
                tmp = self.params[key].flat[i]

                self.params[key].flat[i] = tmp + h
                fxh1 = loss_W(self.params[key])
                
                self.params[key].flat[i] = tmp - h 
                fxh2 = loss_W(self.params[key])
                
                grads[key].flat[i] = (fxh1 - fxh2) / (2*h)
                self.params[key].flat[i] = tmp 

        return grads

    def gradient(self, x, t):
        # Forward
        self.loss(x, t)
        
        # Backward
        dout = 1
        dout = self.last_layer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        # Setting gradients
        grads = {}
        grads['w1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['w2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads

    def save_params(self, file_name="params.pkl"):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            self.params = pickle.load(f)
        self.update_layers()