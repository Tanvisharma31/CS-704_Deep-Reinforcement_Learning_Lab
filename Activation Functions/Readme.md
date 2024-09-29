In this experiment, we’ll compare three common activation functions—ReLU, Sigmoid, and Tanh—by using them in a simple neural network and observing their behavior.

### Common Activation Functions

1. **ReLU (Rectified Linear Unit)**:
   \[
   f(x) = \max(0, x)
   \]
   - Output is zero if the input is negative, otherwise equal to the input.
   - It’s commonly used in hidden layers of neural networks because of its simplicity and ability to mitigate vanishing gradient problems.

2. **Sigmoid**:
   \[
   f(x) = \frac{1}{1 + e^{-x}}
   \]
   - The output is a value between 0 and 1.
   - It's often used in binary classification problems.

3. **Tanh (Hyperbolic Tangent)**:
   \[
   f(x) = \tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
   \]
   - Output is between -1 and 1, making it centered around 0.
   - Used in hidden layers of neural networks for faster convergence.

### Objective:
We'll implement a simple feedforward neural network and compare how each activation function influences the output on the same dataset.

### Step-by-Step Implementation:

```python
import numpy as np
import matplotlib.pyplot as plt

# Define activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Simple feedforward neural network for one hidden layer
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation_function, activation_derivative):
        # Initialize weights
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative

    def feedforward(self, inputs):
        # Calculate the input to the hidden layer
        self.hidden_input = np.dot(inputs, self.weights_input_hidden)
        self.hidden_output = self.activation_function(self.hidden_input)
        
        # Calculate the output layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output)
        self.output = self.activation_function(self.output_input)
        return self.output

    def train(self, inputs, targets, learning_rate=0.1, epochs=10000):
        for epoch in range(epochs):
            # Feedforward
            output = self.feedforward(inputs)
            
            # Calculate the error
            error = targets - output
            
            # Backpropagation (Gradient Descent)
            output_error_term = error * self.activation_derivative(self.output_input)
            hidden_error_term = np.dot(output_error_term, self.weights_hidden_output.T) * self.activation_derivative(self.hidden_input)
            
            # Update weights
            self.weights_hidden_output += np.dot(self.hidden_output.T, output_error_term) * learning_rate
            self.weights_input_hidden += np.dot(inputs.T, hidden_error_term) * learning_rate

# Generate a simple dataset (XOR problem)
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])  # XOR output

# Compare different activation functions
activations = {
    "Sigmoid": (sigmoid, sigmoid_derivative),
    "Tanh": (tanh, tanh_derivative),
    "ReLU": (relu, relu_derivative)
}

# Training the neural networks with different activations
results = {}

for name, (activation, derivative) in activations.items():
    print(f"Training with {name} activation function")
    
    # Initialize neural network
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1,
                             activation_function=activation, activation_derivative=derivative)
    
    # Train the network
    nn.train(inputs, targets, learning_rate=0.1, epochs=10000)
    
    # Test the network
    predictions = nn.feedforward(inputs)
    results[name] = predictions

# Plot the results
plt.figure(figsize=(12, 6))
for i, (name, output) in enumerate(results.items()):
    plt.subplot(1, 3, i + 1)
    plt.bar(range(4), output.flatten(), color='blue')
    plt.ylim(0, 1)
    plt.title(f'{name} Output')
    plt.xticks(range(4), ['[0,0]', '[0,1]', '[1,0]', '[1,1]'])
plt.tight_layout()
plt.show()
```

### Explanation:
- **Activation Functions**: We define `sigmoid`, `tanh`, and `relu` along with their respective derivatives.
- **Feedforward Neural Network**: The network has one hidden layer with 4 neurons. The chosen activation function is applied to both the hidden layer and output layer.
- **Training**: The network is trained using backpropagation with gradient descent for 10,000 epochs on the XOR problem.
- **Plotting**: The final outputs of the neural network for the four input combinations are plotted for comparison.

### Observations:
- **Sigmoid**: Produces outputs between 0 and 1, but may struggle with values close to 0 or 1 because of vanishing gradients.
- **Tanh**: Centered around 0, allowing better convergence for problems like XOR, but still faces issues with vanishing gradients.
- **ReLU**: May perform well, especially in deeper networks due to its ability to avoid the vanishing gradient problem. However, it can suffer from "dead neurons" (neurons stuck with 0 output) if inputs are negative for a prolonged period.

You can observe the difference in the outputs for these activation functions and how they handle the XOR problem.