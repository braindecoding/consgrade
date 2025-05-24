class AttentionLayer(nn.Module):
    """Attention mechanism for LSTM"""
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: [batch_size, seq_len, hidden_size]
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        # attention_weights shape: [batch_size, seq_len, 1]
        
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        # context_vector shape: [batch_size, hidden_size]
        
        return context_vector, attention_weights

class BidirectionalLSTMWithWavelet(nn.Module):
    """Bidirectional LSTM with attention and wavelet features"""
    def __init__(self, input_size, hidden_size, wavelet_features_dim, num_layers=2, dropout_rate=0.5):
        super(BidirectionalLSTMWithWavelet, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # Bidirectional LSTM for raw EEG data
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_size * 2)  # *2 for bidirectional
        
        # Fully connected layers for wavelet features
        self.fc_wavelet1 = nn.Linear(wavelet_features_dim, hidden_size)
        self.batch_norm_wavelet = nn.BatchNorm1d(hidden_size)
        
        # Combined fully connected layers
        self.fc_combined = nn.Linear(hidden_size * 2 + hidden_size, hidden_size)  # LSTM + wavelet
        self.fc_out = nn.Linear(hidden_size, 2)  # 2 classes
    
    def forward(self, x_raw, x_wavelet):
        # Process raw EEG data with LSTM
        # x_raw shape: [batch_size, seq_len, input_size]
        lstm_output, _ = self.lstm(x_raw)
        # lstm_output shape: [batch_size, seq_len, hidden_size*2]
        
        # Apply attention
        context_vector, _ = self.attention(lstm_output)
        # context_vector shape: [batch_size, hidden_size*2]
        
        # Process wavelet features
        x_wavelet = self.fc_wavelet1(x_wavelet)
        x_wavelet = self.batch_norm_wavelet(x_wavelet)
        x_wavelet = F.elu(x_wavelet)
        x_wavelet = F.dropout(x_wavelet, self.dropout_rate, training=self.training)
        
        # Combine features
        x_combined = torch.cat((context_vector, x_wavelet), dim=1)
        
        # Final classification
        x_combined = self.fc_combined(x_combined)
        x_combined = F.elu(x_combined)
        x_combined = F.dropout(x_combined, self.dropout_rate, training=self.training)
        x_combined = self.fc_out(x_combined)
        
        return x_combined