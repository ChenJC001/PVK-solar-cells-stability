import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import shap

data = pd.read_excel(r'C:\Users\Administrator\Desktop\shuju\quxian_shiyan.xlsx', sheet_name='>500', header=None)
feature = data.iloc[:, :76]
target = data.iloc[:, 76:]
x = feature
y = target
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1412)
scaler = MinMaxScaler()
scaler.fit(X_train)
result_x_train = scaler.transform(X_train)  # Transform training data
result_x_test = scaler.transform(X_test)    # Transform test data
scaler2 = MinMaxScaler()
scaler2.fit(y_train)
result_y_train = scaler2.transform(y_train)  # Transform training data
result_y_test = scaler2.transform(y_test)    # Transform test data

# Define neural network model
class CurvePredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation, l1_lambda, l2_lambda):
        super(CurvePredictor, self).__init__()
        self.layers = nn.ModuleList()
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        # Input layer to first hidden layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(get_activation(activation))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.layers.append(get_activation(activation))

        # Last hidden layer to output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def l1_loss(self):
        l1_reg = torch.tensor(0.0, dtype=torch.float32)
        for param in self.parameters():
            l1_reg += torch.norm(param, 1)  # Use L1 norm to calculate regularization term
        return self.l1_lambda * l1_reg

    def l2_loss(self):
        l2_reg = torch.tensor(0.0, dtype=torch.float32)
        for param in self.parameters():
            l2_reg += torch.norm(param, 2)
        return self.l2_lambda * l2_reg

def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f"Invalid activation function: {activation}")

def custom_loss(outputs, targets, lambda_denorm=0.1):
    # Calculate regular prediction error loss
    loss = nn.MSELoss()(outputs, targets)

    # Denormalize model outputs and true target values
    denorm_outputs = scaler2.inverse_transform(outputs.detach().numpy())
    denorm_targets = scaler2.inverse_transform(targets.detach().numpy())

    # Calculate mean squared error after denormalization as regularization term
    denorm_loss = nn.MSELoss()(torch.from_numpy(denorm_outputs), torch.from_numpy(denorm_targets))

    # Add regular loss and regularization term
    total_loss = loss + lambda_denorm * denorm_loss

    return total_loss

# Set parameters
input_size = result_x_train.shape[1]  # Number of input features
output_size = result_y_train.shape[1]  # Output dimension
learning_rate = 0.001
num_epochs = 2000
l2_lambda = 0.00000001
l1_lambda = 0.000000008
lambda_denorm = 0.0001

# Instantiate model
model = CurvePredictor(input_size=input_size, hidden_sizes=[512, 256, 256, 128, 64], output_size=output_size
                       , activation='relu', l1_lambda=l1_lambda, l2_lambda=l2_lambda)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Record losses during training
train_losses = []
test_losses = []

# Set early stopping parameters
patience = 300  # Stop training if test loss does not improve for this many epochs
min_delta = 0.00001  # Only consider it an improvement if test loss is less than the current minimum by at least this much

# Record the best model and corresponding test loss
best_model = None
best_test_loss = float('inf')
patience_counter = 0

# Training loop
for epoch in range(num_epochs):
    # Get input data and target values
    inputs = torch.from_numpy(result_x_train).float()
    targets = torch.from_numpy(result_y_train).float()

    # Forward pass
    outputs = model(inputs)
    loss = custom_loss(outputs, targets, lambda_denorm=lambda_denorm) + model.l1_loss() + model.l2_loss()  # Add L1 and L2 regularization terms

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Record loss
    train_losses.append(loss.item())

    # Print loss
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate model and record test loss
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(result_x_test).float()
        y_test_tensor = torch.from_numpy(result_y_test).float()
        test_outputs = model(X_test_tensor)
        test_loss = custom_loss(test_outputs, y_test_tensor, lambda_denorm=lambda_denorm) + model.l2_loss()  # Add L2 regularization term
        # Check if the best model and corresponding test loss need to be updated
        if test_loss.item() < best_test_loss - min_delta:
            best_test_loss = test_loss.item()
            best_model = model
            patience_counter = 0
        else:
            patience_counter += 1

        # If patience is exhausted, stop training
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

        test_losses.append(test_loss.item())

# Print final test loss
print(f'Final Test Loss: {test_loss.item():.4f}')

# Load the best model
model = best_model

# Save the model
torch.save(model, 'quxian_high500.pth')

# Plot training and test loss curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Load the entire model
model = torch.load('quxian_high500.pth')

# Make predictions
X_test_tensor = torch.from_numpy(result_x_test).float()
predictions = model(X_test_tensor)

# Denormalize the prediction results
predictions_numpy = predictions.detach().numpy()
pred_denorm = scaler2.inverse_transform(predictions_numpy)

# Prepare the true data for comparison
targets = scaler2.inverse_transform(result_y_test)

# Set the row indices to plot
indices = list(range(1, 37))

# Create a canvas with 100 subplots
fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(20, 16))
axes = axes.ravel()

# Iterate over the indices and plot the curves
for i, index in enumerate(indices):
    true_curve = targets[index]
    pred_curve = pred_denorm[index].ravel()  # Extract denormalized prediction values
    axes[i].plot(pred_curve[20:], pred_curve[:20], label='Prediction curve', color="red", linewidth=2.5, linestyle="-")
    axes[i].plot(true_curve[20:], true_curve[:20], label='True curve', color="blue", linewidth=2.5, linestyle="-")
    axes[i].set_xlabel('Time (h)')
    axes[i].set_ylabel('Normalized PCE')
    axes[i].set_title(f'Time vs Efficiency (Row {index})')
    axes[i].set_ylim(0, 1.2)
    rmse = np.sqrt(mean_squared_error(true_curve, pred_curve))
    print(f'RMSE: {rmse:.4f}')

plt.tight_layout()
plt.show()