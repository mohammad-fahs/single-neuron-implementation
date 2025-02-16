# Implementing Gradient Descent for a Single Neuron with MSE

# Loading and Exploring the dataset

Using Pandas library we will read the network CSV file into a Data Frame (`df`) and then display general information about the dataset

```python
import pandas as pd

# Load the dataset
file_path = "Social_Network_Ads.csv"
df = pd.read_csv(file_path)

# Display basic information and first few rows
df.info(), df.head()
```

according to the Info, the dataset contains 400 entries with three columns `Age`, `EstimatedSalary` and `Purchased` . All columns have non-null values so we don’t need to handle any missing data. 

```
RangeIndex: 400 entries, 0 to 399
Data columns (total 3 columns):
 #   Column           Non-Null Count  Dtype
---  ------           --------------  -----
 0   Age              400 non-null    int64
 1   EstimatedSalary  400 non-null    int64
 2   Purchased        400 non-null    int64
dtypes: int64(3)

 Age  EstimatedSalary  Purchased
 0   19            19000          0
 1   35            20000          0
 2   26            43000          0
 3   27            57000          0
 4   19            76000          0
```

# Splitting Dataset for training and testing

split the dataset into training and testing sets. I'll also normalize the features to improve training efficiency. 80% for training and 20% for testing 

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target variable
X = df[['Age', 'EstimatedSalary']].values
y = df['Purchased'].values

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Verify the shape of the splits
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```

the shape of the test and train sets are as below 

- **`X_train.shape`**: (320, 2) → 320 training samples, 2 features.
- **`X_test.shape`**: (80, 2) → 80 test samples, 2 features.
- **`y_train.shape`**: (320,) → 320 target values for training.
- **`y_test.shape`**: (80,) → 80 target values for testing

→ **`StandardScaler`** This class is used for feature scaling (standardization) to ensure that all features have a mean of 0 and a standard deviation of 1.

- `fit`: Computes the mean and standard deviation from `X_train`.
- `transform`: Applies the transformation to standardize the training data.

Features like `Age` (which ranges from 18 to 60) and `EstimatedSalary` (which ranges from 15,000 to 150,000) have different scales. Gradient Descent converges faster when features are on a similar scale and normalization prevents features with large values from dominating smaller values features 

# **Neuron Network Implementation**

## Main Concepts:

### Weights

Weights are the **parameters** of a neural network that determine how much influence an input feature has on the final prediction.  When training a neural network, these weights are adjusted iteratively to minimize the error, If the weight is large, the corresponding input has a **strong impact** on the prediction  If the weight is small or close to zero, the feature has **less impact** on the output.
In our case, since we have two input features (`*Age`* and `*EstimatedSalary*`), we have two weights.

### Bias

Bias is an additional **constant value** added to the output of the neuron. It helps the model **shift the decision boundary** so it can better fit the data.  If the bias is **zero**, the output is strictly dependent on the input values.  If the bias is **nonzero**, it allows the neuron to make a prediction even when all input features are zero. Without bias, the model might struggle to fit the dataset correctly.

### Gradient

A **gradient** is a vector that points in the direction of the **steepest increase** of a function. it helps us determine how to update model parameters (weights and bias) to minimize the loss function.

In neural networks, we aim to **minimize the loss function** (e.g., Mean Squared Error). The **gradient** tells us the direction and magnitude of change required for each parameter (weights and bias) to reduce the loss.

### Learning rate

The **learning rate**(α) controls how much the weights and bias are adjusted during training. A **high learning rate** makes big changes to the weights, which might cause the model to overshoot the optimal values and fail to converge.  A **low learning rate** makes small adjustments, which leads to slow learning but better accuracy.  The learning rate needs to be **carefully chosen** for efficient training. gradient represents how much the weight needs to be adjusted.

### Epochs

An **epoch** is **one complete pass** through the entire training dataset. Training a model requires multiple epochs to adjust the weights correctly if you use **too few epochs**, the model might not learn enough patterns (underfitting) and if you use **too many epochs**, the model might learn noise instead of patterns (overfitting)

## Code

### Initialize Parameters

The function `np.random.randn()` generates random numbers, which can be different every time the code runs. By setting a fixed seed (`42` in this case), we ensure that the same random numbers (weights and bias) are generated every time. weights are initialized randomly for model with 2 inputs, 
If all weights were initialized to **zero**, the model wouldn’t learn effectively because all neurons would behave the same way (symmetry problem), random values ensure **diversity** in learning.
Similar to weights, initializing bias with a random value ensures diversity in the learning process. 
The bias allows the model to **shift** predictions up or down, helping it fit the data better.

learning rate defines the step size for updating weights and bias in **gradient descent**.
epochs defines how many times the model will iterate over the training data.

```python
# Initialize parameters (weights and bias)
np.random.seed(42)
weights = np.random.randn(2)  # Two features
bias = np.random.randn()

# Define the learning rate and number of epochs
learning_rate = 0.01
epochs = 1000
```

### Define Loss Function (Mean Squared Error)

```python
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
```

### Training Loop

```python
# Training loop
for epoch in range(epochs):
    # Forward propagation: Compute predictions
    y_pred = np.dot(X_train, weights) + bias
    
    # Compute loss
    loss = mse_loss(y_train, y_pred)
    
    # Compute gradients
    error = y_pred - y_train
    dW = np.dot(X_train.T, error) / len(X_train)  # Gradient w.r.t weights
    dB = np.mean(error)  # Gradient w.r.t bias
    
    # Update weights and bias using gradient descent
    weights -= learning_rate * dW
    bias -= learning_rate * dB

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

```

The above loop runs 1000 iterations to optimize wight and bias, each time in this loop a prediction is computed using the linear operation and the loss for the prediction is also calculated. after computing the the prediction and loss value gradient is computed and according to the derivation of `dW` (gradient w.r.t weights) and derivation of `dB` (gradient w.r.t Bias) weight and bias are updated.

### Print  final Weights and Bias

```python
# Final weights and bias after training
weights, bias
```

## Result:

The initial loss at epoch 0 is 0.3311, the loss decreases significantly in the first 100 epochs to 0.1627 and then the loss stabilizes around 0.1307 after 500 epochs indicating convergence . 
After `Epoch 500`, the loss does not significantly change, meaning the model has reached **optimal weights** for minimizing error.

```
Epoch 0, Loss: 0.3311
Epoch 100, Loss: 0.1627
Epoch 200, Loss: 0.1359
Epoch 300, Loss: 0.1315
Epoch 400, Loss: 0.1308
Epoch 500, Loss: 0.1307
Epoch 600, Loss: 0.1307
Epoch 700, Loss: 0.1307
Epoch 800, Loss: 0.1307
Epoch 900, Loss: 0.1307
(array([0.26017436, 0.14529949]), np.float64(0.3593874468550853))
```

⇒ the final function is 

```
 Y’ = (0.2601 * X1) + (0.1453 * X2) + 0.3594
```

# Test Loss

In machine learning, **test loss** refers to the measure of how well the model is performing on the test dataset, which consists of data that the model has not seen during training. It helps to understand how accurately the model generalizes to new, unseen data.
The model makes predictions on the test data (`X_test`) using the learned parameters `weights` and `bias`. This is done using the formula `y_test_pred = np.dot(X_test, weights) + bias`, which is a linear regression prediction.
The **test loss** is calculated using **Mean Squared Error (MSE)**, a common loss function used to measure the difference between the true values (`y_test`) and the predicted values (`y_test_pred`).

The value `0.0993` for **Test Loss** means that, on average, the squared difference between the true values and the predicted values is approximately `0.0993`. Lower test loss values indicate that the model is doing a better job in predicting the true values, and higher values indicate worse predictions. In this case, the model is performing decently. 

```python
# Make predictions on the test set
y_test_pred = np.dot(X_test, weights) + bias

# Compute test loss
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

test_loss = mse_loss(y_test, y_test_pred)
print("Test Loss:", test_loss)
```