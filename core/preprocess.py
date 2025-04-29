import os
import shutil
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse

# Load config.yaml
def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Hàm để xoá và tạo mới thư mục lưu dữ liệu tiền xử lý
def reset_preprocessed_dirs(preprocessed_normal_dir, preprocessed_abnormal_dir):
    for dir_path in [preprocessed_normal_dir, preprocessed_abnormal_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"Deleted existing directory: {dir_path}")
        os.makedirs(dir_path)
        print(f"Created fresh directory: {dir_path}")

# Hàm để lưu các Mel-spectrogram từ file âm thanh, chia thành các chunk
def save_mel_spectrogram_chunks(wav_file, output_dir, filename_prefix, config):
    sample_rate = config['preprocessing']['sample_rate']
    chunk_size = config['preprocessing']['chunk_size']
    n_fft = config['preprocessing']['mel_n_fft']
    win_length = config['preprocessing']['mel_win_length']
    hop_length = config['preprocessing']['mel_hop_length']
    n_mels = config['preprocessing']['mel_num_mels']

    # Load audio với sample_rate chỉ định
    y, sr = librosa.load(wav_file, sr=sample_rate)
    total_duration = librosa.get_duration(y=y, sr=sr)
    samples_per_chunk = int(chunk_size * sr)
    num_chunks = int(np.ceil(len(y) / samples_per_chunk))

    print(f"\nProcessing {os.path.basename(wav_file)}: {total_duration:.2f}s → {num_chunks} chunks")

    for i in range(num_chunks):
        start_sample = i * samples_per_chunk
        end_sample = start_sample + samples_per_chunk
        chunk = y[start_sample:end_sample]

        if len(chunk) < samples_per_chunk:
            print(f"Skipping last short chunk ({len(chunk)/sr:.2f}s)")
            break

        mel_spec = librosa.feature.melspectrogram(
            y=chunk,
            sr=sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels
        )

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize mel spectrogram to [0,1]
        mel_spec_norm = (mel_spec_db + 80) / 80
        mel_spec_norm = np.clip(mel_spec_norm, 0, 1)

        # Plot and save
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            mel_spec_norm,
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='mel',
            cmap='magma'
        )
        plt.colorbar(format='%+2.0f dB')

        output_filename = f"{filename_prefix}_chunk{i}.png"
        output_path = os.path.join(output_dir, output_filename)

        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"Saved chunk {i}: {output_path}")

# Hàm tiền xử lý dữ liệu âm thanh (normal + abnormal)
def preprocess_data(config_path):
    config = load_config(config_path)
    paths = config['data']

    normal_dir = paths['normal_dir']
    abnormal_dir = paths['abnormal_dir']
    preprocessed_normal_dir = paths['preprocessed_normal_dir']
    preprocessed_abnormal_dir = paths['preprocessed_abnormal_dir']

    # Xóa và tạo lại thư mục tiền xử lý
    reset_preprocessed_dirs(preprocessed_normal_dir, preprocessed_abnormal_dir)

    # Tiền xử lý cho thư mục normal
    print("\n=== Processing NORMAL files ===")
    for wav_file in os.listdir(normal_dir):
        if wav_file.endswith(".wav"):
            wav_path = os.path.join(normal_dir, wav_file)
            filename_prefix = wav_file.replace(".wav", "")
            save_mel_spectrogram_chunks(wav_path, preprocessed_normal_dir, filename_prefix, config)

    # Tiền xử lý cho thư mục abnormal
    print("\n=== Processing ABNORMAL files ===")
    for wav_file in os.listdir(abnormal_dir):
        if wav_file.endswith(".wav"):
            wav_path = os.path.join(abnormal_dir, wav_file)
            filename_prefix = wav_file.replace(".wav", "")
            save_mel_spectrogram_chunks(wav_path, preprocessed_abnormal_dir, filename_prefix, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config.yaml')
    args = parser.parse_args()

    preprocess_data(args.config)
