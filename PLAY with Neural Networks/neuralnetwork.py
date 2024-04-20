#Q1.Creating the Neural Network with numpy
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# Importing the required libraries

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network architecture
input_size = X_train.shape[1]
hidden_size = 10
output_size = len(np.unique(y_train))

# Initialize the weights and biases
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros(hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros(output_size)

# Define the activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the forward propagation function
def forward_propagation(X):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return A2

# Define the training loop
learning_rate = 0.01
num_iterations = 1000

for i in range(num_iterations):
    # Forward propagation
    A2 = forward_propagation(X_train)
    
    # Compute the loss
    loss = -np.mean(y_train * np.log(A2) + (1 - y_train) * np.log(1 - A2))
    
    # Backpropagation
    dZ2 = A2 - y_train
    dW2 = np.dot(A1.T, dZ2) / len(X_train)
    db2 = np.mean(dZ2, axis=0)
    dZ1 = np.dot(dZ2, W2.T) * A1 * (1 - A1)
    dW1 = np.dot(X_train.T, dZ1) / len(X_train)
    db1 = np.mean(dZ1, axis=0)
    
    # Update the weights and biases
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

# Evaluate the model on the test set
predictions = forward_propagation(X_test)
predictions = np.argmax(predictions, axis=1)
accuracy = np.mean(predictions == y_test) * 100
print(f"Accuracy: {accuracy}%")
# Plot the loss curve
plt.plot(range(num_iterations), loss_curve)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()