import os
import tensorflow as tf
from core.data_loader import create_dataset
from core.model.vqvae import build_vqvae
import yaml
import matplotlib.pyplot as plt

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def plot_loss(history, save_path):
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    # Load config
    config = load_config()
    paths = config['paths']
    training_cfg = config['training']
    model_cfg = config['model']

    batch_size = training_cfg['batch_size']
    epochs = training_cfg['epochs']

    # Create datasets
    train_ds, val_ds = create_dataset(mode='train')

    # Build model
    model = build_vqvae(
        input_shape=model_cfg['input_shape'],
        embedding_dim=model_cfg['embedding_dim'],
        num_embeddings=model_cfg['num_embeddings'],
        commitment_cost=model_cfg['commitment_cost']
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')

    # Prepare checkpoint
    checkpoint_dir = paths['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, 'vqvae_best.h5')

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    # Start training
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint_cb]
    )

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, 'vqvae_final.h5')
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Plot and save loss curve
    loss_plot_path = os.path.join(checkpoint_dir, 'loss_curve.png')
    plot_loss(history, loss_plot_path)
    print(f"Loss curve saved to {loss_plot_path}")

if __name__ == "__main__":
    main()
