In this experiment, we’ll explore various regularization techniques on an autoencoder, including **L2 regularization**, **dropout**, and **noise injection**, to understand their effects on the model’s ability to generalize and reduce overfitting.

### Experiment Overview

1. **Data Preparation**:
   - Load the MNIST dataset and normalize it.
   
2. **Autoencoder Baseline**:
   - Build a simple autoencoder model as the baseline.
   
3. **Regularization Techniques**:
   - Add L2 regularization to the weights of the encoder and decoder.
   - Introduce dropout layers in the encoder.
   - Inject noise into the input data to create a denoising autoencoder.
   
4. **Training and Comparison**:
   - Train the autoencoder with each technique and compare reconstruction quality on test data.
   - Observe how each technique affects the loss and reconstruction quality.

### Code Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GaussianNoise
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Load and preprocess the MNIST dataset
(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

# Function to create a baseline autoencoder model
def create_autoencoder(l2_reg=0, dropout_rate=0, noise_factor=0):
    input_img = Input(shape=(784,))
    
    # Optional noise injection layer
    if noise_factor > 0:
        noisy_input = GaussianNoise(noise_factor)(input_img)
    else:
        noisy_input = input_img
    
    # Encoder with optional L2 regularization and dropout
    encoded = Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(noisy_input)
    if dropout_rate > 0:
        encoded = Dropout(dropout_rate)(encoded)
    
    encoded = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(encoded)
    encoded_output = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(encoded)
    
    # Decoder with optional L2 regularization
    decoded = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(encoded_output)
    decoded = Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(decoded)
    decoded_output = Dense(784, activation='sigmoid')(decoded)
    
    autoencoder = Model(input_img, decoded_output)
    return autoencoder

# Function to train and evaluate autoencoder
def train_autoencoder(autoencoder, epochs=10):
    autoencoder.compile(optimizer=Adam(), loss='mse')
    history = autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=256, shuffle=True, validation_data=(X_test, X_test))
    return history

# Define different regularization techniques
techniques = {
    'Baseline': (0, 0, 0),
    'L2 Regularization': (0.001, 0, 0),
    'Dropout': (0, 0.3, 0),
    'Noise Injection': (0, 0, 0.3)
}

# Train autoencoders with each regularization technique and record history
histories = {}
for technique, (l2_reg, dropout_rate, noise_factor) in techniques.items():
    print(f"\nTraining with {technique}...")
    autoencoder = create_autoencoder(l2_reg=l2_reg, dropout_rate=dropout_rate, noise_factor=noise_factor)
    history = train_autoencoder(autoencoder, epochs=10)
    histories[technique] = history.history['val_loss']

# Plot validation loss for comparison
plt.figure(figsize=(10, 6))
for technique, val_loss in histories.items():
    plt.plot(val_loss, label=technique)
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.title('Validation Loss Across Different Regularization Techniques')
plt.show()

# Visualize reconstruction quality on test data for each technique
def visualize_reconstructions(technique):
    autoencoder = create_autoencoder(*techniques[technique])
    autoencoder.compile(optimizer=Adam(), loss='mse')
    autoencoder.fit(X_train, X_train, epochs=10, batch_size=256, shuffle=True)
    
    decoded_imgs = autoencoder.predict(X_test[:10])
    plt.figure(figsize=(15, 4))
    for i in range(10):
        # Display original
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        # Display reconstruction
        ax = plt.subplot(2, 10, i + 1 + 10)
        plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.suptitle(f"Reconstruction with {technique}")
    plt.show()

# Visualize reconstructions for each technique
for technique in techniques.keys():
    visualize_reconstructions(technique)
```

### Explanation

1. **Autoencoder Configurations**:
   - The `create_autoencoder` function builds an autoencoder with options for **L2 regularization**, **dropout**, and **noise injection**.

2. **Training and Evaluation**:
   - Each autoencoder is trained for 10 epochs, and **validation loss** (MSE) is recorded for comparison.

3. **Visualization**:
   - We plot the validation loss for each technique to see how the regularization affects model generalization.
   - After training, the reconstruction quality of each technique is visualized on a subset of test images.

### Expected Results

- **Baseline**: May overfit quickly, leading to higher validation loss.
- **L2 Regularization**: Penalizes large weights, potentially improving generalization with slightly higher reconstruction error but lower overfitting.
- **Dropout**: Reduces reliance on any single neuron, resulting in better generalization but potentially slower learning.
- **Noise Injection**: Forces the network to learn noise-robust features, likely leading to better generalization on noisy data but may slightly increase training difficulty.

This experiment highlights the trade-offs of regularization techniques in autoencoders, where each technique’s benefits and effects depend on the network’s goal and the dataset's characteristics.