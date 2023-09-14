import numpy as np
import matplotlib.pyplot as plt

# sigmoid activation function for the hidden layer
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# linear activation function for the output layer
def linear(x):
    return x

def main():

    # 1. initialize weights randomly
    # 2. forward propagation to get an output result
    # 3. find value of cost (see how good the output is)
    # 4. back propagation to fix weight and biases
    # 5. repeat steps 2,3,4 until cost function is minimized

#   ==============================================================================================
#   ==============================================================================================

    # define network
    np.random.seed()
    input_neurons = 2
    hidden_neurons = 3
    output_neurons = 1

    # initialize weights and biases for the hidden layer
    weights_hidden = np.random.randn(input_neurons, hidden_neurons)
    bias_hidden = np.random.randn(hidden_neurons)

    # initialize weights and biases for the output layer
    weights_output = np.random.randn(hidden_neurons, output_neurons)
    bias_output = np.random.randn(output_neurons)

    # XOR training set
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    learning_rate = 0.1
    epochs = 2000
    loss_history = []

    # training
    for epoch in range(epochs):

        # forward pass
        hidden_input = bias_hidden + np.dot(X, weights_hidden)
        hidden_output = sigmoid(hidden_input)
        output_input = bias_output + np.dot(hidden_output, weights_output)
        output = linear(output_input)

        # get loss for plotting purposes
        loss = np.mean((output - y) ** 2)
        loss_history.append(loss)

        # backward pass with gradient descent
        output_error = output - y  # MSE
        hidden_error = output_error.dot(weights_output.T)
        
        # get the gradients
        output_delta = output_error  # Linear activation derivative is 1
        hidden_delta = hidden_error * (hidden_output * (1 - hidden_output))  # Took the derivative of gradient
        
        # update the weights and biases, finish backpropagation 
        weights_output -= learning_rate * hidden_output.T.dot(output_delta)
        bias_output -= learning_rate * np.sum(output_delta, axis=0)
        weights_hidden -= learning_rate * X.T.dot(hidden_delta)
        bias_hidden -= learning_rate * np.sum(hidden_delta, axis=0)

    # test model on XOR
    test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    predicted_output = linear(sigmoid(test_input.dot(weights_hidden) + bias_hidden).dot(weights_output) + bias_output)
    print("Predicted Output:")
    print(predicted_output)
  
    # create plot
    plt.figure(figsize=(12, 4))
    
    # plot loss function
    plt.subplot(1, 3, 1)
    plt.plot(range(epochs), loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')

    # plot test_input
    plt.subplot(1, 3, 2)
    plt.scatter(test_input[:, 0], test_input[:, 1], c='black', marker='o', label='Input Data')
    plt.xlabel('X input')
    plt.ylabel('Y input')
    plt.title('Input Dataset')

    # plot predicted results
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