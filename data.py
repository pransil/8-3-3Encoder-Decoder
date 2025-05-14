import torch
import random
# Set random seed for reproducibility
random.seed(42)

# Generate n_samples of 'one-hot' data, each data_dim wide
# Also generate test data, all possible values for this data_dim
def generate_data(data_dim, n_samples):
    # Training data
    train = torch.zeros((n_samples, data_dim))  # Initialize with all zeros
    for i in range(n_samples):
        pos = random.randint(0, data_dim-1)  # Random position for the 1
        train[i, pos] = 1  # Set that position to 1
    
    # Test data (all possible inputs)
    test = torch.zeros(data_dim, data_dim)
    for i in range(data_dim):
        test[i, i] = 1
    print("Data generation complete!")
    print(f"train: {train.shape}")
    print(f"test: {test.shape}")

    return train, test

