# Consumer Grade EEG Source Code

Implementasi kode terdiri dari beberapa script Python:

1. simple_analysis.py: Implementasi awal dengan model sederhana
2. best_model.py: Implementasi SVM dengan fitur wavelet
3. wavelet_visualization.py: Visualisasi analisis wavelet
4. eeg_pytorch.py: Implementasi CNN dengan PyTorch
5. eeg_pytorch_improved.py: Implementasi CNN yang ditingkatkan
6. eeg_wavelet_cnn.py: Implementasi CNN dengan fitur wavelet
7. eeg_lstm_wavelet.py: Implementasi BiLSTM dengan fitur wavelet
8. eeg_transformer.py: Implementasi Transformer dengan fitur wavelet
9. compare_models.py: Perbandingan semua model

=======================================================================
LAMPIRAN: INSTRUKSI UNTUK MEREPRODUKSI PENELITIAN
=======================================================================
Implementasi memerlukan library Python berikut:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- PyWavelets
- PyTorch

A. Persiapan Lingkungan
----------------------
1. Instal Python 3.8 atau yang lebih baru
2. Instal library yang diperlukan:
   ```
   pip install numpy pandas scikit-learn matplotlib seaborn pywavelets torch
   ```

B. Struktur Dataset
------------------
Dataset harus memiliki format berikut:
- File teks dengan kolom yang dipisahkan tab
- Column 5: Digit (6 or 9)
- Column 7: Data EEG (comma-separated values)

C. Langkah-langkah Reproduksi
----------------------------
1. Clone repository atau download semua script Python
2. Letakkan dataset di folder "Data" dengan nama "EP1.01.txt"
3. Jalankan script untuk model yang ingin direproduksi:
   ```
   python best_model.py           # SVM with Wavelet
   python eeg_wavelet_cnn.py      # CNN with Wavelet
   python eeg_lstm_wavelet.py     # BiLSTM with Attention
   python eeg_transformer.py      # Transformer
   ```
4. Untuk membandingkan semua model:
   ```
   python simple_comparison.py
   ```

D. Output yang Diharapkan
------------------------
1. Akurasi dan metrik evaluasi lainnya untuk setiap model
2. Plot training history
3. Confusion matrices
4. Visualisasi wavelet (jika menggunakan wavelet_visualization.py)

E. Troubleshooting
-----------------
1. Jika terjadi error "Cannot convert numpy.ndarray to numpy.ndarray", pastikan 
   versi numpy dan scikit-learn kompatibel
2. Jika terjadi error CUDA, pastikan PyTorch diinstal dengan dukungan CUDA yang 
   sesuai atau gunakan CPU mode
3. Jika terjadi error memory, kurangi batch size atau jumlah data yang digunakan
