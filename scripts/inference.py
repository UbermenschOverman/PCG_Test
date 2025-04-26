import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import yaml
from core.preprocess import audio_to_mel

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_audio(filepath, config):
    sr = config['preprocessing']['sample_rate']
    duration = config['preprocessing']['chunk_duration']

    audio, _ = librosa.load(filepath, sr=sr)

    # Nếu file dài hơn 1 chunk -> lấy đúng 1 đoạn
    if len(audio) > sr * duration:
        audio = audio[:sr * duration]

    # Nếu file ngắn hơn 1 chunk -> pad
    if len(audio) < sr * duration:
        padding = sr * duration - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')

    return audio

def preprocess_audio(audio, config):
    mel = audio_to_mel(
        audio,
        sr=config['preprocessing']['sample_rate'],
        n_mels=config['preprocessing']['n_mels'],
        hop_length=config['preprocessing']['hop_length'],
        n_fft=config['preprocessing']['n_fft']
    )

    mel = (mel - np.min(mel)) / (np.max(mel) - np.min(mel) + 1e-8)  # normalize
    mel = mel[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)

    return mel

def plot_compare(input_mel, output_mel, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(8,4))

    axes[0].imshow(input_mel.squeeze(), cmap='gammatone')
    axes[0].set_title('Input Mel')
    axes[0].axis('off')

    axes[1].imshow(output_mel.squeeze(), cmap='gammatone')
    axes[1].set_title('Reconstructed Mel')
    axes[1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved inference result to {save_path}")
    else:
        plt.show()
    plt.close()

def main(audio_path):
    # Load config
    config = load_config()
    paths = config['paths']

    # Load model
    checkpoint_path = os.path.join(paths['checkpoint_dir'], 'vqvae_best.h5')
    model = tf.keras.models.load_model(checkpoint_path, compile=False)
    print(f"Loaded model from {checkpoint_path}")

    # Load and preprocess audio
    audio = load_audio(audio_path, config)
    input_mel = preprocess_audio(audio, config)

    # Predict
    output_mel = model.predict(input_mel)
    reconstruction_error = np.mean((input_mel - output_mel) ** 2)
    print(f"\nReconstruction MSE: {reconstruction_error:.6f}")

    # Save result
    save_dir = paths['result_dir']
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    save_path = os.path.join(save_dir, f"{filename}_inference.png")

    plot_compare(input_mel, output_mel, save_path=save_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=str, required=True, help="Path to input .wav file")
    args = parser.parse_args()

    main(args.audio)
