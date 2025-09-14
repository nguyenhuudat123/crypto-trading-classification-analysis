"""
PyTorch Model Definitions for Bitcoin Price Prediction
Contains LSTM, GRU, and CNN models for time series classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    """LSTM model for sequence classification"""
    def __init__(self, input_size, hidden_size=50, num_layers=2, num_classes=4, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, num_classes)
        
    def forward(self, x):
        device = x.device
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take last output
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class GRUModel(nn.Module):
    """GRU model for sequence classification"""
    def __init__(self, input_size, hidden_size=50, num_layers=2, num_classes=4, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, num_classes)
        
    def forward(self, x):
        device = x.device
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        
        # GRU forward pass
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])  # Take last output
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class CNNModel(nn.Module):
    """1D CNN model for sequence classification"""
    def __init__(self, input_size, sequence_length=10, num_classes=4):
        super(CNNModel, self).__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = self._calculate_conv_output_size()
        self.fc1 = nn.Linear(conv_output_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        
    def _calculate_conv_output_size(self):
        """Calculate output size after convolutions"""
        # After conv1 (padding=1): sequence_length
        # After conv2 (padding=1): sequence_length
        # After maxpool (kernel_size=2): sequence_length // 2
        pooled_length = self.sequence_length // 2
        return 64 * pooled_length
        
    def forward(self, x):
        # Reshape for 1D CNN: (batch, channels, sequence)
        x = x.transpose(1, 2)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def build_lstm_model(input_size, device='cpu'):
    """Create PyTorch LSTM model"""
    model = LSTMModel(input_size=input_size)
    return model.to(device)

def build_gru_model(input_size, device='cpu'):
    """Create PyTorch GRU model"""
    model = GRUModel(input_size=input_size)
    return model.to(device)

def build_cnn_model(input_size, sequence_length=10, device='cpu'):
    """Create PyTorch CNN model"""
    model = CNNModel(input_size=input_size, sequence_length=sequence_length)
    return model.to(device)