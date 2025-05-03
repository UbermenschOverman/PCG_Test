import os
import shutil
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pywt
from scipy.signal import resample
import soundfile as sf
import argparse

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def reset_preprocessed_dirs(preprocessed_normal_dir, preprocessed_abnormal_dir):
    for dir_path in [preprocessed_normal_dir, preprocessed_abnormal_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"Deleted existing directory: {dir_path}")
        os.makedirs(dir_path)
        print(f"Created fresh directory: {dir_path}")

def pad_or_trim_to_fixed_length(y, sr, target_duration=12.5):
    target_samples = int(target_duration * sr)
    total_samples = len(y)

    if total_samples >= target_samples:
        return y[-target_samples:]
    else:
        shortage = target_samples - total_samples
        head_repeat = int(np.ceil(shortage / total_samples))
        y_padded = np.tile(y, head_repeat + 1)
        return np.concatenate((y, y_padded[:shortage]))

def stft_custom(signal, fs, nperseg=64, noverlap=32, window='hamming'):
    if window == 'hamming':
        win = np.hamming(nperseg)
    else:
        raise ValueError("Unsupported window type")

    step = nperseg - noverlap
    num_windows = (len(signal) - noverlap) // step
    t = np.arange(num_windows) * step / fs

    Zxx = np.zeros((nperseg // 2 + 1, num_windows), dtype=complex)
    for i in range(num_windows):
        start_idx = i * step
        end_idx = start_idx + nperseg
        if end_idx > len(signal): break
        segment = signal[start_idx:end_idx] * win
        Zxx[:, i] = np.fft.rfft(segment)
    f = np.fft.rfftfreq(nperseg, d=1 / fs)
    return f, t, Zxx

def process_wav_and_save(wav_path, output_path, config, filename_prefix):
    sample_rate = config['preprocessing']['sample_rate']
    y, sr = librosa.load(wav_path, sr=sample_rate)
    y_fixed = pad_or_trim_to_fixed_length(y, sr)

    # Chuẩn hóa về [-1, 1]
    y_fixed = 2 * (y_fixed - np.min(y_fixed)) / (np.max(y_fixed) - np.min(y_fixed)) - 1

    # Wavelet + Upsample
    coeffs = pywt.wavedec(y_fixed, wavelet='db4', level=3)
    cA = coeffs[0]
    upsampled = pywt.upcoef('a', cA, wavelet='db4', level=3, take=len(y_fixed))

    # Downsample về 1000Hz
    new_len = int(len(upsampled) * 1000 / sr)
    downsampled = resample(upsampled, new_len)

    # STFT
    f, t_, Zxx = stft_custom(downsampled, 1000, nperseg=64, noverlap=32)
    S_mag = np.abs(Zxx)
    S_mag = (S_mag - np.min(S_mag)) / (np.max(S_mag) - np.min(S_mag) + 1e-10)

    # Mel
    mel_filterbank = librosa.filters.mel(sr=1000, n_fft=64, n_mels=8, fmin=10, fmax=250)
    mel_output = np.dot(mel_filterbank, S_mag)

    # Lưu ảnh
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_output, sr=1000, hop_length=32,
                             x_axis='time', y_axis='mel', cmap='cool', fmin=10, fmax=250)
    plt.colorbar(label='Mel-Scaled Power')
    plt.title(f'Mel Spectrogram: {filename_prefix}')
    output_img_path = os.path.join(output_path, f"{filename_prefix}.png")
    plt.savefig(output_img_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved: {output_img_path}")

def split_and_move_files(source_dir, train_dir, val_dir, test_dir, split_ratio=(0.7, 0.2, 0.1)):
    all_files = [f for f in os.listdir(source_dir) if f.endswith(".png")]
    np.random.shuffle(all_files)
    
    total = len(all_files)
    train_end = int(split_ratio[0] * total)
    val_end = train_end + int(split_ratio[1] * total)
    
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    for target_dir in [train_dir, val_dir, test_dir]:
        os.makedirs(target_dir, exist_ok=True)

    for file_list, dest in [(train_files, train_dir), (val_files, val_dir), (test_files, test_dir)]:
        for file in file_list:
            src = os.path.join(source_dir, file)
            dst = os.path.join(dest, file)
            shutil.move(src, dst)
            print(f"Moved {file} → {dest}")

def preprocess_data(config_path):
    config = load_config(config_path)
    paths = config['data']

    # Đường dẫn xử lý trung gian
    preprocessed_temp_normal_dir = "preprocessed_data/temp_normal"
    preprocessed_temp_abnormal_dir = "preprocessed_data/temp_abnormal"

    reset_preprocessed_dirs(preprocessed_temp_normal_dir, preprocessed_temp_abnormal_dir)

    print("\n=== Processing NORMAL files ===")
    for wav_file in os.listdir(paths['normal_dir']):
        if wav_file.endswith(".wav"):
            wav_path = os.path.join(paths['normal_dir'], wav_file)
            filename_prefix = wav_file.replace(".wav", "")
            process_wav_and_save(wav_path, preprocessed_temp_normal_dir, config, filename_prefix)

    print("\n=== Processing ABNORMAL files ===")
    for wav_file in os.listdir(paths['abnormal_dir']):
        if wav_file.endswith(".wav"):
            wav_path = os.path.join(paths['abnormal_dir'], wav_file)
            filename_prefix = wav_file.replace(".wav", "")
            process_wav_and_save(wav_path, preprocessed_temp_abnormal_dir, config, filename_prefix)

    print("\n=== Splitting NORMAL files ===")
    split_and_move_files(
        preprocessed_temp_normal_dir,
        paths['preprocessed_train_normal_dir'],
        paths['preprocessed_val_normal_dir'],
        paths['preprocessed_test_normal_dir']
    )

    print("\n=== Splitting ABNORMAL files ===")
    split_and_move_files(
        preprocessed_temp_abnormal_dir,
        paths['preprocessed_train_abnormal_dir'],
        paths['preprocessed_val_abnormal_dir'],
        paths['preprocessed_test_abnormal_dir']
    )

    # Dọn thư mục tạm
    shutil.rmtree(preprocessed_temp_normal_dir)
    shutil.rmtree(preprocessed_temp_abnormal_dir)
    print("\n✅ Preprocessing and splitting completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config.yaml')
    args = parser.parse_args()
    preprocess_data(args.config)
