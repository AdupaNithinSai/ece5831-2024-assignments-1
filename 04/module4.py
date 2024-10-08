import numpy as np
from multilayer_perceptron import MultiLayerPerceptron

def main():
    """
    Main function to test the MultiLayerPerceptron class.
    
    Initializes an instance of the MultiLayerPerceptron, trains it on a set of dummy data,
    and then makes predictions on test data.
    """
    # Create an instance of MultiLayerPerceptron
    mlp = MultiLayerPerceptron()

    # Training data
    x_train = np.array([[0.1, 0.3], [0.4, 0.2], [0.7, 0.9], [0.6, 0.5]])
    y_train = np.array([[0.5, 0.2], [0.1, 0.7], [0.2, 0.5], [0.9, 0.4]])

    # Train the model
    mlp.train(x_train, y_train, epochs=1000)
    
    # Test data point
    x_test = np.array([1.0, 0.5])
    
    # Predict using the trained model
    prediction = mlp.predict(x_test)
    
    print("Prediction of the MultiLayerPerceptron for input [1.0, 0.5]:")
    print(prediction)

    mlp.plot_decision_boundary(mlp, x_train, y_train)



if __name__ == "__main__":
    main()
