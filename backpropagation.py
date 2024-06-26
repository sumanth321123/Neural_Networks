import numpy as np

input_size = 2
hidden_size = 3
output_size = 1

def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights_input_hidden = np.random.randn(input_size, hidden_size)
    biases_hidden = np.zeros((1, hidden_size))
    weights_hidden_output = np.random.randn(hidden_size, output_size)
    biases_output = np.zeros((1, output_size))
    return weights_input_hidden, biases_hidden, weights_hidden_output, biases_output

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)

def forward_propagation(input_data, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output):
    hidden_input = np.dot(input_data, weights_input_hidden) + biases_hidden
    hidden_output = sigmoid(hidden_input)
    output = np.dot(hidden_output, weights_hidden_output) + biases_output
    return hidden_input, hidden_output, output

def backward_propagation(input_data, hidden_input, hidden_output, output, target, weights_hidden_output):
    output_error = mse_loss_derivative(target, output)
    hidden_layer_error = output_error.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_output)
    d_weights_hidden_output = hidden_output.T.dot(output_error)
    d_biases_output = np.sum(output_error, axis=0, keepdims=True)
    d_weights_input_hidden = input_data.T.dot(hidden_layer_error)
    d_biases_hidden = np.sum(hidden_layer_error, axis=0, keepdims=True)
    return d_weights_input_hidden, d_biases_hidden, d_weights_hidden_output, d_biases_output

learning_rate = 0.1
epochs = 100
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

weights_input_hidden, biases_hidden, weights_hidden_output, biases_output = initialize_weights(input_size, hidden_size, output_size)

for epoch in range(epochs):
    hidden_input, hidden_output, network_output = forward_propagation(X, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output)
    loss = mse_loss(y, network_output)
    gradients = backward_propagation(X, hidden_input, hidden_output, network_output, y, weights_hidden_output)
    d_weights_input_hidden, d_biases_hidden, d_weights_hidden_output, d_biases_output = gradients
    weights_input_hidden -= learning_rate * d_weights_input_hidden
    biases_hidden -= learning_rate * d_biases_hidden
    weights_hidden_output -= learning_rate * d_weights_hidden_output
    biases_output -= learning_rate * d_biases_output
    
    if epoch % 1 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')
        _, _, predictions = forward_propagation(X, weights_input_hidden, biases_hidden, weights_hidden_output, biases_output)
        correct_predictions = (predictions.round() == y)
        accuracy = np.mean(correct_predictions) * 100
        print("Predictions: ", predictions.round())
        print("Accuracy: ", accuracy)
