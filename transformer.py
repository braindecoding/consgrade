class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    def __init__(self, d_model, max_seq_length=1000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerWithWavelet(nn.Module):
    """Transformer model with wavelet features for EEG classification"""
    def __init__(self, input_size, d_model, nhead, num_layers, wavelet_features_dim, dropout_rate=0.1):
        super(TransformerWithWavelet, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fully connected layers for wavelet features
        self.fc_wavelet1 = nn.Linear(wavelet_features_dim, d_model)
        self.batch_norm_wavelet = nn.BatchNorm1d(d_model)
        
        # Combined fully connected layers
        self.fc_combined = nn.Linear(d_model * 2, d_model)  # Transformer + wavelet
        self.fc_out = nn.Linear(d_model, 2)  # 2 classes
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x_raw, x_wavelet):
        # Process raw EEG data with Transformer
        # x_raw shape: [batch_size, seq_len, input_size]
        
        # Project input to d_model dimensions
        x_projected = self.input_projection(x_raw)
        
        # Add positional encoding
        x_projected = self.positional_encoding(x_projected)
        
        # Apply Transformer encoder
        transformer_output = self.transformer_encoder(x_projected)
        
        # Global average pooling over sequence dimension
        transformer_output = torch.mean(transformer_output, dim=1)
        
        # Process wavelet features
        x_wavelet = self.fc_wavelet1(x_wavelet)
        x_wavelet = self.batch_norm_wavelet(x_wavelet)
        x_wavelet = F.elu(x_wavelet)
        x_wavelet = self.dropout(x_wavelet)
        
        # Combine features
        x_combined = torch.cat((transformer_output, x_wavelet), dim=1)
        
        # Final classification
        x_combined = self.fc_combined(x_combined)
        x_combined = F.elu(x_combined)
        x_combined = self.dropout(x_combined)
        x_combined = self.fc_out(x_combined)
        
        return x_combined