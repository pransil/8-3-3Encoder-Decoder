# 838.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from test import test_model, create_confusion_matrix, write_confusion_matrix_to_file
from data import generate_data
data_dim = 8
hidden_dim = 2
data_items = 1000

# Generate training data
train, test = generate_data(data_dim, n_samples=data_items)

inputs = train
# Define the model
class BinaryAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.Sigmoid()  # Compresses to 3-unit representation
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, data_dim),
            nn.Sigmoid()  # Outputs probabilities for 8-bit reconstruction
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize model, loss, and optimizer
model = BinaryAutoencoder()
criterion = nn.BCELoss()  # Measures pixel-wise binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Training loop
for epoch in range(4000):
    outputs = model(inputs)
    loss = criterion(outputs, inputs)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

accuracy, confusion_matrix = test_model(model, test)