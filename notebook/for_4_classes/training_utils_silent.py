"""
Training Utilities for Bitcoin Price Prediction
Handles model training, evaluation, and loss calculation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score
from pytorch_models import build_lstm_model, build_gru_model, build_cnn_model
from xgboost_model import build_xgboost_model

def train_pytorch_model(model, X_train, y_train, X_test, y_test, epochs=30, batch_size=32, verbose=False):
    """
    Train PyTorch model and return predictions
    
    Args:
        model: PyTorch model to train
        X_train, y_train: Training data
        X_test, y_test: Test data
        epochs: Number of training epochs
        batch_size: Batch size for training
        verbose: Whether to print training progress
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if verbose and (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, y_pred = torch.max(test_outputs, 1)
        y_pred = y_pred.cpu().numpy()
        y_test_cpu = y_test.cpu().numpy()
    
    return y_pred, y_test_cpu

def calculate_directional_loss(y_true, y_pred, price_changes):
    """
    Calculate loss based on wrong directional predictions for 4-class classification
    
    Args:
        y_true: True labels (0: strong decline, 1: slight decline, 2: slight rise, 3: strong rise)
        y_pred: Predicted labels
        price_changes: Actual price changes
    
    Returns:
        loss_count: Number of wrong direction predictions
        loss_mean: Average loss of wrong predictions
    """
    loss_values = np.zeros(len(y_true))
    
    # Loss when predicting wrong direction for extreme cases
    # Predicting strong decline (0) when price actually goes up (positive change)
    # Predicting strong rise (3) when price actually goes down (negative change)
    wrong_predictions = ((y_pred == 0) & (price_changes > 0)) | ((y_pred == 3) & (price_changes < 0))
    loss_values[wrong_predictions] = np.abs(price_changes[wrong_predictions])
    
    transaction = np.sum((y_pred == 0) | (y_pred == 3))

    loss_count = np.sum(loss_values > 0)
    loss_mean = np.mean(loss_values[loss_values > 0]) if loss_count > 0 else 0
    
    return loss_count, loss_mean, transaction

def train_xgboost_model(X_train, X_test, y_train, y_test, df_processed, test_indices):
    """Train and evaluate XGBoost model"""
    model = build_xgboost_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    price_changes = df_processed.loc[test_indices, 'price_change'].values
    loss_count, loss_mean, transaction = calculate_directional_loss(y_test, y_pred, price_changes)
    
    return accuracy, loss_count, loss_mean, transaction

def train_pytorch_model_wrapper(model_type, X_train, X_test, y_train, y_test, df_processed, 
                               device='cpu', sequence_length=10, verbose=False):
    """
    Train and evaluate PyTorch models
    
    Args:
        model_type: Type of model ('LSTM', 'GRU', 'CNN')
        X_train, X_test, y_train, y_test: Data tensors
        df_processed: Processed dataframe for loss calculation
        device: Device to run on
        sequence_length: Length of input sequences
        verbose: Whether to print training progress
    """
    input_size = X_train.shape[2]
    
    # Select model type
    if model_type == 'LSTM':
        model = build_lstm_model(input_size, device)
    elif model_type == 'GRU':
        model = build_gru_model(input_size, device)
    elif model_type == 'CNN':
        model = build_cnn_model(input_size, sequence_length, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train and get predictions
    y_pred, y_test_cpu = train_pytorch_model(model, X_train, y_train, X_test, y_test, verbose=verbose)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_cpu, y_pred)
    
    # Get corresponding price changes for loss calculation
    test_start_idx = len(df_processed) - len(y_test_cpu)
    price_changes = df_processed['price_change'].iloc[test_start_idx:test_start_idx + len(y_test_cpu)].values
    loss_count, loss_mean, transaction = calculate_directional_loss(y_test_cpu, y_pred, price_changes)
    
    return accuracy, loss_count, loss_mean, transaction