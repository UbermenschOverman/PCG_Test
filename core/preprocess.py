import os
import librosa
import numpy as np
import tensorflow as tf
import yaml
import matplotlib.pyplot as plt

# Load config.yaml
def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Hàm để lưu Mel-spectrogram thành file ảnh PNG
def save_mel_spectrogram(wav_file, output_dir, filename):
    y, sr = librosa.load(wav_file, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)  # Cập nhật cách gọi hàm melspectrogram
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Save mel spectrogram as an image
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Hàm tiền xử lý dữ liệu âm thanh
def preprocess_data(config_path='config/config.yaml'):
    config = load_config(config_path)
    paths = config['data']

    normal_dir = paths['normal_dir']
    abnormal_dir = paths['abnormal_dir']
    preprocessed_normal_dir = paths['preprocessed_normal_dir']
    preprocessed_abnormal_dir = paths['preprocessed_abnormal_dir']

    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(preprocessed_normal_dir, exist_ok=True)
    os.makedirs(preprocessed_abnormal_dir, exist_ok=True)

    # Tiền xử lý cho thư mục normal
    for wav_file in os.listdir(normal_dir):
        if wav_file.endswith(".wav"):
            wav_path = os.path.join(normal_dir, wav_file)
            output_filename = wav_file.replace(".wav", ".png")
            save_mel_spectrogram(wav_path, preprocessed_normal_dir, output_filename)
    
    # Tiền xử lý cho thư mục abnormal
    for wav_file in os.listdir(abnormal_dir):
        if wav_file.endswith(".wav"):
            wav_path = os.path.join(abnormal_dir, wav_file)
            output_filename = wav_file.replace(".wav", ".png")
            save_mel_spectrogram(wav_path, preprocessed_abnormal_dir, output_filename)

if __name__ == "__main__":
    preprocess_data()
