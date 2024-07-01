import torch

# Load the model weights from the .pth file
weights = torch.load("mnist_model_weights.pth")

# Print the keys in the dictionary to see the layer names
print("Keys in the state_dict:")
for key in weights.keys():
    print(key)

# Optionally, print out the weights for a specific layer
# Example: print the weights of the first convolutional layer
conv1_weights = weights["conv1.weight"]
print("\nconv1 weights:")
print(conv1_weights)

# Print the shape of the weights for each layer
print("\nShapes of the weights for each layer:")
for key, value in weights.items():
    print(f"{key}: {value.shape}")

# For a detailed inspection, you might want to use a debugger or an interactive environment like Jupyter Notebook
