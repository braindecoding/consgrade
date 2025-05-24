def load_digits_simple(file_path, target_digits=[6, 9], max_per_digit=500):
    """Load EEG data for digit classification"""
    print(f"📂 Loading data for digits {target_digits}...")
    
    if file_path is None or not os.path.exists(file_path):
        print("❌ Dataset file not found!")
        return None, None
    
    # Initialize data containers
    data_6 = []
    data_9 = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
                
            # Split by TAB
            parts = line.split('\t')
            
            # Need at least 7 columns
            if len(parts) < 7:
                continue
            
            try:
                # Column 5 (index 4) = digit
                digit = int(parts[4])
                
                # Only process if it's in target digits
                if digit in target_digits:
                    # Column 7 (index 6) = data
                    data_string = parts[6]
                    
                    # Parse comma-separated values
                    values = [np.float64(x.strip()) for x in data_string.split(',') if x.strip()]
                    
                    # Store based on digit
                    if digit == 6 and len(data_6) < max_per_digit:
                        data_6.append(values)
                    elif digit == 9 and len(data_9) < max_per_digit:
                        data_9.append(values)
            except (ValueError, IndexError):
                continue
    
    # Combine data and labels
    all_data = data_6 + data_9
    all_labels = [0] * len(data_6) + [1] * len(data_9)  # 0 for digit 6, 1 for digit 9
    
    # Normalize lengths
    normalized_data = []
    target_length = 1792  # 14 channels * 128 timepoints
    
    for trial in all_data:
        if len(trial) >= target_length:
            # Truncate if too long
            normalized_data.append(trial[:target_length])
        else:
            # Pad with repetition if too short
            trial_copy = trial.copy()
            while len(trial_copy) < target_length:
                trial_copy.extend(trial[:min(len(trial), target_length - len(trial_copy))])
            normalized_data.append(trial_copy[:target_length])
    
    data = np.array(normalized_data, dtype=np.float64)
    labels = np.array(all_labels, dtype=np.int32)
    
    return data, labels

def extract_wavelet_features(data):
    """Extract wavelet features from EEG data"""
    # Reshape data to 14 channels x 128 timepoints
    reshaped_data = []
    for trial in data:
        try:
            # Reshape to 14 x 128
            reshaped = trial.reshape(14, 128)
            reshaped_data.append(reshaped)
        except ValueError:
            continue
    
    # Define wavelet parameters
    wavelet = 'db4'  # Daubechies wavelet with 4 vanishing moments
    level = 4        # Decomposition level
    
    # Extract wavelet features
    wavelet_features = []
    
    for trial in reshaped_data:
        trial_features = []
        
        # Process each channel
        for channel in range(trial.shape[0]):
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(trial[channel], wavelet, level=level)
            
            # Extract features from each level
            for i in range(level + 1):
                # Calculate energy
                energy = np.sum(coeffs[i]**2)
                
                # Calculate entropy
                entropy = -np.sum(coeffs[i]**2 * np.log(coeffs[i]**2 + 1e-10))
                
                # Calculate mean and standard deviation
                mean = np.mean(coeffs[i])
                std = np.std(coeffs[i])
                
                # Add features
                trial_features.extend([energy, entropy, mean, std])
        
        wavelet_features.append(trial_features)
    
    wavelet_features = np.array(wavelet_features, dtype=np.float64)
    
    return wavelet_features, reshaped_data