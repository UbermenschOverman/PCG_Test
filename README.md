# Phonocardiogram Classification using VQ-VAE (VERSION: TEST)

## Project Overview

This project aims to classify **phonocardiogram (PCG) recordings** as either:

- **Normal**: Containing only fundamental heart sounds (S1 and S2)
- **Abnormal**: Containing additional heart sounds (S3, S4) or murmurs

We use a **Vector-Quantized Variational Autoencoder (VQ-VAE)** for unsupervised representation learning. The idea is that the model reconstructs **normal samples** well, but performs worse on **abnormal samples**. Therefore, reconstruction error (MSE) can serve as an indicator for classification.

---

## Objectives

- Convert raw PCG `.wav` files into time-frequency spectrograms via a **custom preprocessing pipeline**
- Train a VQ-VAE model to learn a discrete latent representation of normal PCG signals
- Use **Mean Squared Error (MSE)** as a threshold for classifying unseen data
- Evaluate performance using F1-score, Confusion Matrix, and ROC AUC

---

## Preprocessing Pipeline

Preprocessing is applied to `.wav` files into PNG spectrograms as follows:

Raw .wav audio files are converted into spectrogram images through the following steps:

**1, Padding/Trimming**: All signals are adjusted to a fixed duration (12.5 seconds) for consistency.
**2, Normalization**: Signals are scaled to the range −1,1 for numerical stability.
**3, Denoising via DWT**: A 3-level Discrete Wavelet Transform (DWT) using Daubechies-4 (db4) is applied to extract approximation coefficients (cA).
**4, Upsampling**: Approximation coefficients are upsampled back to original signal length using pywt.upcoef.
**5, Downsampling**: Signal is resampled to 1000 Hz to reduce temporal resolution.
**6, STFT**: Custom Short-Time Fourier Transform (STFT) is applied using Hamming window (64-point FFT, 50% overlap).
**7, Mel Filtering**: The STFT magnitude is filtered with a Mel filterbank (8 mel bands, 10–250 Hz) to emphasize perceptually important frequency content.
**8, Spectrogram Saving**: Resulting mel spectrogram is saved as a PNG image using librosa.display.specshow.

---

## Model Architecture

### Vector Quantized VAE (VQ-VAE)

- **Encoder**: Convolutional layers to reduce spatial resolution
- **Quantizer**: VectorQuantizer layer (custom Keras layer) for discrete latent representation
- **Decoder**: Transposed convolution layers to reconstruct the input

Loss function:
- **Reconstruction Loss (MSE)**
- **Commitment Loss** (for vector quantization)

Model architecture is defined in: core/model/vqvae.py

## Folder structure is shown below:
PCG_Test/
├── config/
│   └── config.yaml
│
├── core/
│   ├── __init__.py          
│   ├── data_loader.py       # load dataset train/test
│   ├── preprocess.py        # xử lý .wav -> .png
│   ├── test.py              # script test model duoc luu tu saved_models
│   ├── train.py             # script training
│   ├── utils.py             # support function (vẽ ảnh, tính MSE, ensure dir)
│   └── model/
│       ├── __init__.py      
│       └── vqvae.py         # kiến trúc VQ-VAE
│
├── data/
│   ├── normal/              # file .wav bình thường
│   └── abnormal/            # file .wav bất thường
│
├── preprocessed_data/       # push len git se khong co directory nay vi no trong khong, vay nen phai create directory nay, cac subdirectory ben trong se duoc preprocess.py tu tao ra
│   ├── test_abnormal
│   ├── test_normal
│   ├── train_normal
│   ├── train_abnormal
│   ├── val_normal              
│   └── val_abnormal            
│
├── results/
│   ├── __init__.py          # results/__init__.py
│   ├── evals/               # kết quả evaluation (test/inference)
│   └── reconstructed/       # ảnh Groundtruth vs Reconstruction
│
├── saved_models/
│   └── vqvae_best.h5        # model đã train
├── scripts/
│   ├── __init__.py          # scripts/__init__.py
│   ├── inference.py         # predict 1 file .wav bất kỳ - barebone version
├── requirement.txt
└── README.md (optional)     # mô tả project

---

## Command lines to run (from directory PCG_Test):
1, python3 core/preprocess.py --config config/config.yaml   # Tien xu ly du lieu tu data => preprocessed_data
2, python3 core/train.py --config config/config.yaml        # Train va luu mo hinh .h5 vao saved_models
3, python3 core/test.py --config config/config.yaml         # Test model da train va luu test_log vao results/evals, Average MSE, F1-score, Confusion matrix, ROC curve

## List of libraries/dependencies required: tensorflow, keras, pandas, numpy, matplotlib, gammatone, yacs, pydub, PyWavelets, opencv, sklearn, librosa.
(Install with: pip install -r requirements.txt)