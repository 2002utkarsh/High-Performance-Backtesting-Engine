import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from typing import Tuple

class TimeSeriesTransformer(nn.Module):
    """
    Transformer Encoder for Time-Series Regression.
    Predicts next-step price (scalar) based on a sequence of features.
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, output_dim=1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.decoder = nn.Linear(d_model, output_dim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_embedding.bias.data.zero_()
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # src shape: [batch_size, seq_len, input_dim]
        src = self.input_embedding(src) # [batch_size, seq_len, d_model]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src) # [batch_size, seq_len, d_model]
        
        # We take the output of the last time step for prediction
        output = output[:, -1, :] # [batch_size, d_model]
        output = self.decoder(output) # [batch_size, output_dim]
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class StockDataset(Dataset):
    def __init__(self, features, targets, seq_len=30):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) # Float for regression
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        return (self.features[idx:idx+self.seq_len], self.targets[idx+self.seq_len-1])

def train_model(model, train_loader, criterion, optimizer, epochs=5, device='cpu'):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            # outputs shape [batch, 1], targets shape [batch] -> unsqueeze targets or squeeze outputs
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}')

def predict(model, data, seq_len=30, device='cpu'):
    model.eval()
    # Dummy targets for prediction phase
    dataset = StockDataset(data, np.zeros(len(data)), seq_len=seq_len) 
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.squeeze().cpu().numpy())
            
    # Pad initial sequence length with NaN or first prediction
    pad = [np.nan] * seq_len
    return np.concatenate([pad, np.array(predictions)])

def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path="model.pth", device="cpu"):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        print(f"Model loaded from {path}")
        return True
    return False


