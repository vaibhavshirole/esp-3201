import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid activation function for the hidden layer
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the linear activation function for the output layer
def linear(x):
    return x

def main():
    # Initialize the weights and biases for the hidden and output layers
    np.random.seed(0)
    input_neurons = 2
    hidden_neurons = 3
    output_neurons = 1

    # Weights and biases for the hidden layer
    weights_hidden = np.random.randn(input_neurons, hidden_neurons)
    bias_hidden = np.random.randn(hidden_neurons)

    # Weights and biases for the output layer
    weights_output = np.random.randn(hidden_neurons, output_neurons)
    bias_output = np.random.randn(output_neurons)

    # Define the XOR training dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Training parameters
    learning_rate = 0.1
    epochs = 2000

    # Lists to store loss values for visualization
    loss_history = []

    # Training the neural network
    for epoch in range(epochs):
        # Forward propagation
        hidden_input = bias_hidden + np.dot(X, weights_hidden)
        hidden_output = sigmoid(hidden_input)
        output_input = bias_output + np.dot(hidden_output, weights_output)
        output = linear(output_input)

        # Calculate the loss (mean squared error)
        loss = np.mean((output - y) ** 2)
        loss_history.append(loss)

        # Backpropagation with gradient descent
        output_error = output - y  # MSE
        hidden_error = output_error.dot(weights_output.T)
        
        # Compute gradients
        output_delta = output_error  # Linear activation derivative is 1
        hidden_delta = hidden_error * (hidden_output * (1 - hidden_output))  # Took the derivative of gradient
        
        # Update weights and biases, finish backpropagation 
        weights_output -= learning_rate * hidden_output.T.dot(output_delta)
        bias_output -= learning_rate * np.sum(output_delta, axis=0)
        weights_hidden -= learning_rate * X.T.dot(hidden_delta)
        bias_hidden -= learning_rate * np.sum(hidden_delta, axis=0)

    # Test the trained model on the XOR dataset
    test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    predicted_output = linear(sigmoid(test_input.dot(weights_hidden) + bias_hidden).dot(weights_output) + bias_output)
    print("Predicted Output:")
    print(predicted_output)

    # Plot the test_input and the corresponding predicted results
    plt.figure(figsize=(12, 4))

    # Plot the loss function
    plt.subplot(1, 3, 1)
    plt.plot(range(epochs), loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')

    # Plot the test_input
    plt.subplot(1, 3, 2)
    plt.scatter(test_input[:, 0], test_input[:, 1], c='black', marker='o', label='Input Data')
    plt.xlabel('X input')
    plt.ylabel('Y input')
    plt.title('Input Dataset')

    # Plot the predicted results
    plt.subplot(1, 3, 3)
    scatter = plt.scatter(test_input[:, 0], test_input[:, 1], c=predicted_output.flatten(), cmap='coolwarm', s=100)
    plt.xlabel('X input')
    plt.ylabel('Y input')
    plt.colorbar(label='Predicted value at X,Y pair')
    plt.title('Predicted Results')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()