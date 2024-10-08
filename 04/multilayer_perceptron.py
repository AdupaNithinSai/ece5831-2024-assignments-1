import numpy as np
import matplotlib as plt

class MultiLayerPerceptron:
    def __init__(self):
        """
        Initializes the MultiLayerPerceptron instance with an empty network and default learning rate.
        """
        self.net = {}
        self.learning_rate = 0.1
        self.init_network()

    def init_network(self):
        """
        Initializes the network with predefined weights and biases for each layer.
        """
        net = {}
        # layer 1
        net['w1'] = np.array([[0.7, 0.9, 0.3], [0.5, 0.4, 0.1]], dtype=float)
        net['b1'] = np.array([1, 1, 1], dtype=float)
        # layer 2
        net['w2'] = np.array([[0.2, 0.3], [0.4, 0.5], [0.22, 0.1234]], dtype=float)
        net['b2'] = np.array([0.5, 0.5], dtype=float)
        # layer 3 <-- output
        net['w3'] = np.array([[0.7, 0.1], [0.123, 0.314]], dtype=float)
        net['b3'] = np.array([0.1, 0.2], dtype=float)
        
        self.net = net

    def forward(self, x):
        """
        Performs a forward pass through the network and returns the output.
        
        Parameters:
        x (numpy.array): Input data point.
        
        Returns:
        numpy.array: Output of the network after the forward pass.
        """
        w1, w2, w3 = self.net['w1'], self.net['w2'], self.net['w3']
        b1, b2, b3 = self.net['b1'], self.net['b2'], self.net['b3']

        # layer 1
        a1 = np.dot(x, w1) + b1
        z1 = self.sigmoid(a1)
        
        # layer 2
        a2 = np.dot(z1, w2) + b2
        z2 = self.sigmoid(a2)
        
        # layer 3
        a3 = np.dot(z2, w3) + b3
        y = self.identity(a3)

        # Storing the activations for backward pass
        self.cache = {'x': x, 'a1': a1, 'z1': z1, 'a2': a2, 'z2': z2, 'a3': a3, 'y': y}
        return y

    def backward(self, y_true):
        """
        Performs a backward pass through the network using backpropagation to update weights and biases.
        
        Parameters:
        y_true (numpy.array): True labels for the input data.
        
        Returns:
        numpy.array: The calculated error between the predictions and the true labels.
        """
        # Extract cached values
        x, a1, z1, a2, z2, a3, y = self.cache['x'], self.cache['a1'], self.cache['z1'], self.cache['a2'], self.cache['z2'], self.cache['a3'], self.cache['y']
        w1, w2, w3 = self.net['w1'], self.net['w2'], self.net['w3']
        
        # Calculate output error (using Mean Squared Error loss)
        error = y - y_true
        
        # Output layer gradients
        d_w3 = np.dot(z2.reshape(-1, 1), error.reshape(1, -1))
        d_b3 = error
        
        # Hidden layer 2 gradients
        error_hidden2 = np.dot(error, w3.T) * self.sigmoid_derivative(z2)
        d_w2 = np.dot(z1.reshape(-1, 1), error_hidden2.reshape(1, -1))
        d_b2 = error_hidden2
        
        # Hidden layer 1 gradients
        error_hidden1 = np.dot(error_hidden2, w2.T) * self.sigmoid_derivative(z1)
        d_w1 = np.dot(x.reshape(-1, 1), error_hidden1.reshape(1, -1))
        d_b1 = error_hidden1
        
        # Update weights and biases
        self.net['w3'] -= self.learning_rate * d_w3
        self.net['b3'] -= self.learning_rate * d_b3
        self.net['w2'] -= self.learning_rate * d_w2
        self.net['b2'] -= self.learning_rate * d_b2
        self.net['w1'] -= self.learning_rate * d_w1
        self.net['b1'] -= self.learning_rate * d_b1

        return error

    def train(self, x_train, y_train, epochs=1000):
        """
        Trains the model using the provided training data for a specified number of epochs.
        
        Parameters:
        x_train (numpy.array): Training input data.
        y_train (numpy.array): Training true labels.
        epochs (int): Number of training iterations (default is 1000).
        """
        for epoch in range(epochs):
            loss = 0
            for x, y in zip(x_train, y_train):
                y_pred = self.forward(x)
                loss += np.mean((y_pred - y) ** 2)
                self.backward(y)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss/len(x_train)}')

    def predict(self, x):
        """
        Predicts the output for the given input data using the trained model.
        
        Parameters:
        x (numpy.array): Input data point.
        
        Returns:
        numpy.array: Predicted output.
        """
        return self.forward(x)

    def identity(self, x):
        """
        Identity activation function. Simply returns the input.
        
        Parameters:
        x (numpy.array): Input data.
        
        Returns:
        numpy.array: Output data (same as input).
        """
        return x

    def sigmoid(self, x):
        """
        Sigmoid activation function.
        
        Parameters:
        x (numpy.array): Input data.
        
        Returns:
        numpy.array: Output after applying sigmoid function.
        """
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """
        Derivative of the sigmoid function.
        
        Parameters:
        x (numpy.array): Input data (assumed to be output of sigmoid function).
        
        Returns:
        numpy.array: Derivative of sigmoid function.
        """
        return x * (1 - x)
    
    def accuracy(self, y_true, y_pred):
        """
        Computes the accuracy of the predictions.

        Parameters:
        y_true (numpy.array): True labels.
        y_pred (numpy.array): Predicted labels.

        Returns:
        float: Accuracy score.
        """
        return np.sum(y_true == y_pred) / len(y_true)
