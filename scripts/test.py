import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from core.data_loader import create_dataset
from core.model.vqvae import build_vqvae
import yaml

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def plot_reconstruction(inputs, outputs, save_dir, num_samples=5):
    os.makedirs(save_dir, exist_ok=True)

    inputs = inputs.numpy()
    outputs = outputs.numpy()

    for i in range(min(num_samples, inputs.shape[0])):
        fig, axes = plt.subplots(1, 2, figsize=(8,4))
        
        axes[0].imshow(inputs[i].squeeze(), cmap='gammatone')
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')

        axes[1].imshow(outputs[i].squeeze(), cmap='gammatone')
        axes[1].set_title('Reconstruction')
        axes[1].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"sample_{i}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved reconstruction: {save_path}")

def main():
    # Load config
    config = load_config()
    paths = config['paths']
    model_cfg = config['model']

    # Load test dataset
    _, test_ds = create_dataset(mode='test')

    # Load trained model
    checkpoint_path = os.path.join(paths['checkpoint_dir'], 'vqvae_best.h5')
    model = tf.keras.models.load_model(checkpoint_path, compile=False)
    print(f"Loaded model from {checkpoint_path}")

    # Evaluate
    reconstructions = []
    ground_truths = []

    for batch in test_ds:
        recon = model(batch, training=False)
        reconstructions.append(recon)
        ground_truths.append(batch)

    reconstructions = tf.concat(reconstructions, axis=0)
    ground_truths = tf.concat(ground_truths, axis=0)

    # Compute loss
    mse = tf.reduce_mean(tf.square(reconstructions - ground_truths)).numpy()
    print(f"\nReconstruction MSE on test set: {mse:.6f}\n")

    # Save reconstruction samples
    results_dir = paths['result_dir']
    plot_reconstruction(ground_truths, reconstructions, save_dir=results_dir)

if __name__ == "__main__":
    main()
