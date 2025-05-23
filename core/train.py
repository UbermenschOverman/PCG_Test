import os
import tensorflow as tf
from data_loader import create_dataset
from model.vqvae import build_vqvae
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
    paths = config['saved_models']
    training_cfg = config['training']
    model_cfg = config['model']

    batch_size = training_cfg['batch_size']
    epochs = training_cfg['epochs']

    # Tạo dataset với img_size đã được truyền vào
    input_shape = tuple(model_cfg['input_shape'])  # [H, W, C] từ config.yaml
    img_size = input_shape[:2]  # chỉ lấy (H, W)
    train_ds, val_ds = create_dataset(img_size=img_size, mode='train')
    train_ds = train_ds.map(lambda x, _: (x, x))
    val_ds = val_ds.map(lambda x, _: (x, x))

    # Kiểm tra dữ liệu đầu vào
    for images, labels in train_ds.take(1):
      print(f"Train batch shape: {images.shape}, {labels.shape}")

    for images, labels in val_ds.take(1):
      print(f"Val batch shape: {images.shape}, {labels.shape}")

    # Build model
    model = build_vqvae(
        input_shape=input_shape,
        embedding_dim=model_cfg['embedding_dim'],
        num_embeddings=model_cfg['num_embeddings'],
        commitment_cost=training_cfg['commitment_cost']
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')


    # Prepare checkpoint
    checkpoint_dir = paths['model_dir']
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
