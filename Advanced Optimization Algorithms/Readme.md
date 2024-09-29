In this experiment, we'll compare **AdaGrad**, **RMSProp**, and **Adam** optimization algorithms in training a simple neural network. These optimizers are commonly used in deep learning to improve convergence by dynamically adjusting learning rates during training.

### Step-by-Step Plan:

1. **Setup a Neural Network**: We'll create a simple neural network using Keras.
2. **Compare Optimizers**: We will train the same network using AdaGrad, RMSProp, and Adam, and compare their performance.
3. **Visualize the Results**: The training loss and accuracy for each optimizer will be plotted for comparison.

### Neural Network:
We'll use a simple feedforward neural network to classify data from the **MNIST** dataset, which consists of 28x28 grayscale images of handwritten digits (0–9).

### Step-by-Step Code:

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adagrad, RMSprop, Adam

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize the data

# Define a simple feedforward neural network
def create_model(optimizer):
    model = Sequential([
        Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images into a 1D vector
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')  # Output layer for 10 classes
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model using different optimizers
def train_model(optimizer_name, optimizer):
    model = create_model(optimizer)
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=0)
    return history

# Train models with AdaGrad, RMSProp, and Adam optimizers
optimizers = {
    'AdaGrad': Adagrad(),
    'RMSProp': RMSprop(),
    'Adam': Adam()
}

histories = {}
for name, optimizer in optimizers.items():
    print(f"Training with {name} optimizer...")
    histories[name] = train_model(name, optimizer)

# Plot the results: Loss and Accuracy for each optimizer
def plot_histories(histories):
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(history.history['loss'], label=f'{name} - Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(history.history['accuracy'], label=f'{name} - Accuracy')
    plt.title('Training Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_histories(histories)
```

### Explanation:

1. **Loading the Dataset**:
   - We load the **MNIST** dataset using Keras, which is split into training and testing sets.
   - We normalize the data so that the pixel values range between 0 and 1.

2. **Model Architecture**:
   - We define a simple **feedforward neural network** with two hidden layers and ReLU activation. The output layer uses softmax activation for classification.

3. **Optimizers**:
   - **AdaGrad**: This optimizer adapts the learning rate for each parameter, making large updates for infrequent parameters and small updates for frequent parameters.
   - **RMSProp**: RMSProp uses an exponentially weighted moving average of the squared gradients to adjust the learning rate, preventing large oscillations and allowing for steady convergence.
   - **Adam**: Adam combines the benefits of both AdaGrad and RMSProp by using both first and second moments of the gradient, making it one of the most popular optimizers.

4. **Training**:
   - We train the model for 5 epochs using each optimizer and store the training history (loss and accuracy).

5. **Visualization**:
   - We plot the **training loss** and **accuracy** for each optimizer to compare their performance.

### Output:

- The **loss** and **accuracy** curves show how well each optimizer is performing over the epochs.
- Typically, **Adam** optimizer converges faster and achieves better performance than **AdaGrad** and **RMSProp** for many problems, but you may observe different behavior depending on the dataset and network architecture.

### Expected Results:
- **AdaGrad**: May converge more slowly and sometimes perform worse on deeper networks since it keeps decaying the learning rate.
- **RMSProp**: Likely converges faster than AdaGrad due to adaptive learning rates.
- **Adam**: Expected to show faster convergence and better accuracy due to the combination of adaptive learning rates and momentum-based updates.

Feel free to adjust the number of epochs or network architecture to further explore the behavior of each optimizer.