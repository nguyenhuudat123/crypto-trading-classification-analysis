"""
Data Preparation Module for Bitcoin Price Prediction
Handles data loading, technical indicators, normalization, and data preparation
"""

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

def load_bitcoin_data(file_path):
    """Load Bitcoin 1-day OHLCV data"""
    try:
        btc_data = pd.read_csv(file_path)
        print(f"Successfully loaded data from: {file_path}")
        print(f"Data shape: {btc_data.shape}")
        return btc_data.copy()
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def add_technical_indicators(df):
    """Add all technical indicators to the dataframe"""
    df = df.copy()
    prices = df['close']
    volume = df['volume']
    
    # RSI (14-period)
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # EMAs
    df['ema_8'] = prices.ewm(span=8).mean()
    df['ema_34'] = prices.ewm(span=34).mean()
    df['ema_89'] = prices.ewm(span=89).mean()
    
    # MACD
    ema_12 = prices.ewm(span=12).mean()
    ema_26 = prices.ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Stochastic RSI
    rsi_low = df['rsi'].rolling(window=14).min()
    rsi_high = df['rsi'].rolling(window=14).max()
    stoch_rsi = (df['rsi'] - rsi_low) / (rsi_high - rsi_low) * 100
    df['stoch_rsi_k'] = stoch_rsi
    df['stoch_rsi_d'] = stoch_rsi.rolling(window=3).mean()
    
    # Volume indicators
    df['volume_sma_20'] = volume.rolling(window=20).mean()
    df['volume_roc'] = ((volume - volume.shift(10)) / volume.shift(10)) * 100
    
    # On Balance Volume (OBV)
    price_change = prices.diff()
    obv = np.where(price_change > 0, volume, 
                   np.where(price_change < 0, -volume, 0))
    df['obv'] = pd.Series(obv).cumsum()
    
    print("Technical indicators added successfully")
    return df

def normalize_features(df):
    """Normalize technical indicators to 0-1 range"""
    df = df.copy()
    feature_cols = ['rsi', 'ema_8', 'ema_34', 'ema_89', 'macd_signal', 'macd_histogram', 
                   'stoch_rsi_k', 'stoch_rsi_d', 'volume_sma_20', 'volume_roc', 'obv']
    
    for col in feature_cols:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                df[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                df[col] = 0
    
    print("Features normalized successfully")
    return df

def create_target_labels(df, lookahead=5, threshold=0.03):
    """Create 3-class target labels based on future price movement"""
    df = df.copy()
    df['future_price'] = df['close'].shift(-lookahead)
    df['price_change'] = (df['future_price'] - df['close']) / df['close']
    df = df.dropna()
    
    # Create signal: 0=decline, 1=sideways, 2=rise
    df['signal'] = 1  # Default to sideways
    df.loc[df['price_change'] < -threshold, 'signal'] = 0  # Strong decline
    df.loc[df['price_change'] > threshold, 'signal'] = 2   # Strong rise
    
    # Count distribution
    signal_counts = df['signal'].value_counts().sort_index()
    print(f"Signal distribution - Decline(0): {signal_counts.get(0, 0)}, "
          f"Sideways(1): {signal_counts.get(1, 0)}, Rise(2): {signal_counts.get(2, 0)}")
    
    return df

def prepare_traditional_ml_data(df):
    """Prepare data for XGBoost (no sequences needed)"""
    FEATURES = ['rsi', 'ema_8', 'ema_34', 'ema_89', 'macd', 'macd_signal', 
               'macd_histogram', 'stoch_rsi_k', 'stoch_rsi_d', 'volume_sma_20', 
               'volume_roc', 'obv']
    
    # Check if all features exist
    available_features = [f for f in FEATURES if f in df.columns]
    if len(available_features) != len(FEATURES):
        missing = set(FEATURES) - set(available_features)
        print(f"Warning: Missing features: {missing}")
    
    X = df[available_features].copy()
    y = df['signal'].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, X_test.index

def create_time_sequences(X, y, sequence_length=10):
    """Create time sequences for deep learning models"""
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length + 1):
        X_seq.append(X[i:(i + sequence_length)])
        y_seq.append(y[i + sequence_length - 1])
    return np.array(X_seq), np.array(y_seq)

def prepare_pytorch_data(df, sequence_length=10, device='cpu'):
    """Prepare sequential data for PyTorch models"""
    FEATURES = ['rsi', 'ema_8', 'ema_34', 'ema_89', 'macd', 'macd_signal', 
               'macd_histogram', 'stoch_rsi_k', 'stoch_rsi_d', 'volume_sma_20', 
               'volume_roc', 'obv']
    
    # Check if all features exist
    available_features = [f for f in FEATURES if f in df.columns]
    if len(available_features) != len(FEATURES):
        missing = set(FEATURES) - set(available_features)
        print(f"Warning: Missing features for PyTorch: {missing}")
    
    X = df[available_features].values
    y = df['signal'].values
    
    X_seq, y_seq = create_time_sequences(X, y, sequence_length)
    
    if len(X_seq) == 0:
        print("Warning: No sequences created - data too short")
        return None, None, None, None
    
    # Split data chronologically (75%/25%)
    split_idx = int(len(X_seq) * 0.75)
    X_train = X_seq[:split_idx]
    X_test = X_seq[split_idx:]
    y_train = y_seq[:split_idx]
    y_test = y_seq[split_idx:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    print(f"PyTorch data prepared - Train: {X_train_tensor.shape}, Test: {X_test_tensor.shape}")
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor