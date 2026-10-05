# Fashion-MNIST Image Classification Using CNN

A deep learning project that classifies grayscale clothing images into 10 categories using a Convolutional Neural Network built with TensorFlow and Keras.

The project covers data exploration, preprocessing, model training, evaluation, prediction, and saving and reloading the trained model.

## Results

The recorded run achieved **91.50% test accuracy** on 10,000 held-out images.

| Metric | Result |
|---|---:|
| Test accuracy | 91.50% |
| Test loss | Approximately 0.24 |
| Macro F1-score | 0.914 |
| Trainable parameters | 421,642 |
| Epochs completed | 11 |
| Best epoch by validation loss | 9 |

Early stopping monitored validation loss with a patience of two epochs and restored the best weights. Results may vary between training runs.

## Technologies

- Python
- TensorFlow and Keras
- NumPy
- Matplotlib
- scikit-learn
- Google Colab

## Dataset

Fashion-MNIST contains 70,000 grayscale images, each measuring **28 × 28 pixels**, across 10 categories.

| Label | Category |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

The original 60,000 training images were split into training and validation sets using a stratified split with `random_state=42`.

| Set | Images | Purpose |
|---|---:|---|
| Training | 48,000 | Learn model parameters |
| Validation | 12,000 | Monitor performance and select the stopping point |
| Test | 10,000 | Evaluate the selected model |

The original training set contains 6,000 images per category.

## Data Exploration and Preprocessing

- Displayed sample images with their category names.
- Inspected array shapes, pixel data types, and pixel ranges.
- Checked category counts and confirmed balanced training classes.
- Checked for NaN values; none were found.
- Converted image arrays to `float32`.
- Normalized pixel values from 0–255 to 0–1.
- Reshaped images to `(28, 28, 1)` to include the grayscale channel dimension.

## Model Architecture

Both convolutional layers use 3 × 3 kernels, ReLU activation, and `padding="same"`.

| Layer | Output shape per image | Parameters |
|---|---|---:|
| Input | 28 × 28 × 1 | 0 |
| Conv2D — 32 filters | 28 × 28 × 32 | 320 |
| MaxPooling2D — 2 × 2 | 14 × 14 × 32 | 0 |
| Conv2D — 64 filters | 14 × 14 × 64 | 18,496 |
| MaxPooling2D — 2 × 2 | 7 × 7 × 64 | 0 |
| Flatten | 3,136 | 0 |
| Dense — ReLU | 128 | 401,536 |
| Dropout — 0.3 | 128 | 0 |
| Dense — Softmax | 10 | 1,290 |
| **Total** | | **421,642** |

The convolutional layers extract patterns, and the Dense layers combine those features to classify each image. The output layer produces a probability distribution over the 10 categories.

## Training

- **Optimizer:** Adam
- **Loss function:** Sparse categorical cross-entropy
- **Metric:** Accuracy
- **Batch size:** 32
- **Maximum epochs:** 20
- **Early stopping:** Monitor `val_loss`, patience of 2, restore best weights
- **Dropout rate:** 30%

Integer category labels were used directly with sparse categorical cross-entropy.

An initial experiment showed increasing validation loss while training loss continued decreasing. Early stopping was added to select weights based on validation performance.

## Evaluation

The notebook includes:

- Training and validation accuracy curves
- Training and validation loss curves
- Held-out test evaluation
- Individual image predictions
- A confusion matrix
- A classification report with precision, recall, and F1-score

Selected results from the recorded classification report:

| Category | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Trouser | 0.999 | 0.978 | 0.988 |
| Sandal | 0.987 | 0.982 | 0.984 |
| Bag | 0.971 | 0.990 | 0.980 |
| Shirt | 0.778 | 0.707 | 0.741 |

Trouser had the highest F1-score, while Shirt was the most challenging category. The model correctly identified 707 of the 1,000 Shirt images in the test set.

## Running the Notebook

1. Open the project notebook in Google Colab or a Jupyter environment.
2. Install the required packages if they are not already available:

   ```bash
   pip install tensorflow numpy matplotlib scikit-learn
   ```

3. Run the cells in order, starting with dataset loading.
4. Train the model and inspect the evaluation outputs.
5. Update the model save and load paths to match your environment.

The dataset is downloaded automatically on first use. Google Drive paths require Drive to be mounted in Colab. Git authentication setup is not required to train or evaluate the model.

## Saving and Loading

Save the complete trained model:

```python
model.save("fashion_mnist_cnn.keras")
```

Load it later without retraining:

```python
from tensorflow.keras.models import load_model

loaded_model = load_model("fashion_mnist_cnn.keras")
```

The `.keras` file stores the model architecture, learned weights, and optimizer state. The notebook also demonstrates saving weights separately as `fashion_mnist.weights.h5`; this is optional when the complete model is saved.

Predictions from the original and reloaded models were compared on 10 test images using `np.allclose()`, and the comparison returned `True`.

## Prediction Example

After running the notebook’s preprocessing cells:

```python
import numpy as np

# x_test is already normalized and shaped as (10000, 28, 28, 1)
probabilities = loaded_model.predict(x_test[0:1], verbose=0)
predicted_label = np.argmax(probabilities[0])

print("Predicted category:", class_names[predicted_label])
print("Actual category:", class_names[y_test[0]])
```

The model expects normalized input shaped as `(batch_size, 28, 28, 1)`. The `class_names` mapping and external preprocessing steps are not included in the saved model file.

## Limitations

- Evaluation is limited to Fashion-MNIST images.
- Performance on ordinary clothing photographs has not been established.
- Similar clothing categories remain difficult to distinguish.
- Softmax probabilities are model estimates and do not guarantee correct predictions.

## Future Improvements

- Display and analyze misclassified images.
- Explore data augmentation and alternative architectures using validation results.
- Add an interface for trying new images.
- Investigate preprocessing for external clothing photographs.

## Author

Roshin Shibu Thomas
