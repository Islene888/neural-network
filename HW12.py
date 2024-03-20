import numpy as np

# Softmax function definition
def softmax(x):
    exp_shifted = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_shifted / np.sum(exp_shifted, axis=0, keepdims=True)

# Simulate some data
# Let's assume we have 100 samples and 3 features (like 3 different tests or measurements)
np.random.seed(0)  # For reproducibility
data = np.random.randn(3, 100)  # 3 features for 100 samples

# Simulate some labels (for visualization purposes)
true_labels = np.argmax(data, axis=0)  # Just for demonstration, consider the highest feature as 'true' label

# Apply the softmax function to our data
predicted_probabilities = softmax(data)

# Predict labels based on the highest probability from softmax
predicted_labels = np.argmax(predicted_probabilities, axis=0)

# Compare with true labels (in a real scenario, these would come from your dataset)
accuracy = np.mean(predicted_labels == true_labels)
print(f"Accuracy based on simulated data: {accuracy * 100:.2f}%")

# Example usage of softmax results
print("Predicted probability distribution for the first sample:", predicted_probabilities[:, 0])
