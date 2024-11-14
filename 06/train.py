import numpy as np
from two_layer_net_with_back_prop import TwoLayerNetWithBackProp
from mnist_data import MnistData

''' Load the MNIST dataset '''
mnist = MnistData()
(training_data, training_labels), (test_data, test_labels) = mnist.load()

train_images, train_labels = training_data, training_labels
test_images, test_labels = test_data, test_labels

# Hyperparameters
iterations = 10000
batch_size = 16
learning_rate = 0.01
train_size = train_images.shape[0]
iter_per_epoch = int(train_size / batch_size)

network = TwoLayerNetWithBackProp(input_size=784, hidden_size=50, output_size=10)

for i in range(iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = train_images[batch_mask]
    t_batch = train_labels[batch_mask]
    
    grad = network.gradient(x_batch, t_batch)
    for key in ('w1', 'b1', 'w2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    if i % iter_per_epoch == 0:
        train_loss = network.loss(x_batch, t_batch)
        train_acc = network.accuracy(train_images, train_labels)
        test_acc = network.accuracy(test_images, test_labels)
        print(f'iteration: {i}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, test acc: {test_acc:.4f}')

network.save_params("appikonda_mnist_model.pkl")