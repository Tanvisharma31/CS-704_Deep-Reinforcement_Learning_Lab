In this experiment, we’ll implement a **denoising autoencoder**, a type of autoencoder designed to remove noise from data by learning to reconstruct clean outputs from noisy inputs. We'll evaluate its performance on the **MNIST dataset** by adding random noise to images and training the autoencoder to recover the original images.

### Experiment Steps

1. **Data Preparation**:
   - Load and normalize the MNIST dataset.
   - Add random noise to the images to create noisy inputs.
   
2. **Build the Denoising Autoencoder**:
   - Define the encoder to compress the noisy input and a decoder to reconstruct the clean image.
   
3. **Training**:
   - Train the autoencoder with noisy inputs as input and clean images as the target.
   
4. **Evaluation**:
   - Use the trained model to denoise test images and visualize the reconstructed images.

### Code Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GaussianNoise
from tensorflow.keras.optimizers import Adam

# Load and preprocess the MNIST dataset
(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

# Add random noise to the images
noise_factor = 0.5
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape) 
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape) 
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

# Define the Denoising Autoencoder Model
input_img = Input(shape=(784,))
# Encoder
encoded = GaussianNoise(0.2)(input_img)  # additional noise layer for robustness
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded_output = Dense(32, activation='relu')(encoded)

# Decoder
decoded = Dense(64, activation='relu')(encoded_output)
decoded = Dense(128, activation='relu')(decoded)
decoded_output = Dense(784, activation='sigmoid')(decoded)

# Autoencoder Model
autoencoder = Model(input_img, decoded_output)
autoencoder.compile(optimizer=Adam(), loss='mse')

# Train the Autoencoder
autoencoder.fit(X_train_noisy, X_train, epochs=10, batch_size=256, shuffle=True, validation_data=(X_test_noisy, X_test))

# Denoise the test images
denoised_images = autoencoder.predict(X_test_noisy)

# Visualize original noisy and denoised images
n = 10  # number of images to display
plt.figure(figsize=(20, 6))
for i in range(n):
    # Display noisy image
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test_noisy[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    
    # Display original image
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    
    # Display denoised image
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(denoised_images[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

plt.suptitle("Top: Noisy Images | Middle: Original Images | Bottom: Denoised Images")
plt.show()
```

### Explanation

1. **Data Preparation**:
   - We add random Gaussian noise to the MNIST images to create noisy versions of the images (`X_train_noisy` and `X_test_noisy`).
   - The pixel values are clipped to keep them within the range [0, 1].

2. **Autoencoder Architecture**:
   - The **encoder** reduces the 784-dimensional input to a 32-dimensional latent representation, compressing the input data.
   - A **GaussianNoise layer** is added to the encoder, providing robustness by encouraging the model to generalize.
   - The **decoder** reconstructs the image from the compressed representation.

3. **Training**:
   - The autoencoder is trained with noisy inputs as input data and the clean images as the target, using **Mean Squared Error (MSE)** loss.

4. **Evaluation and Visualization**:
   - The trained model is used to denoise the noisy test images.
   - We visualize the noisy input, original image, and the denoised output to assess the model's performance.

### Expected Results and Observations

- **Noisy Images**: The noisy images should show clear pixel-level distortions due to added Gaussian noise.
- **Denoised Images**: The autoencoder should recover most of the original structure, with less noise and smooth details.
- **Comparison**: The effectiveness of the denoising autoencoder can be assessed by comparing the denoised output to the original images, observing how well the network retains essential features while eliminating noise.

This experiment illustrates how denoising autoencoders can be trained to restore original image quality by learning robust representations even with noisy inputs.