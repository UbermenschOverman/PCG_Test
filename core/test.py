import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from data_loader import create_dataset
from model.vqvae import build_vqvae, VectorQuantizer  # Import custom layer
import yaml
from sklearn.metrics import f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from datetime import datetime

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def plot_reconstruction(inputs, outputs, save_dir, num_samples=5):
    os.makedirs(save_dir, exist_ok=True)

    inputs = inputs.numpy()
    outputs = outputs.numpy()

    for i in range(min(num_samples, inputs.shape[0])):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        axes[0].imshow(inputs[i].squeeze(), cmap='inferno')
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')

        axes[1].imshow(outputs[i].squeeze(), cmap='inferno')
        axes[1].set_title('Reconstruction')
        axes[1].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"sample_{i}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved reconstruction: {save_path}")

def save_log(log_text, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"test_log_{timestamp}.txt")
    with open(log_path, 'w') as f:
        f.write(log_text)
    print(f"\nLog saved to: {log_path}\n")

def main():
    # Load config
    config = load_config()
    model_cfg = config['model']
    eval_cfg = config['evaluation']
    paths = config['saved_models']
    results = config['results']

    # Load test dataset
    image_size = tuple(model_cfg['input_shape'][:2])
    test_ds = create_dataset(img_size=image_size, mode='test')

    # Register custom layer VectorQuantizer before loading model
    tf.keras.utils.get_custom_objects()['VectorQuantizer'] = VectorQuantizer

    # Load trained model
    checkpoint_path = os.path.join(paths['model_dir'], paths['model_filename'])
    model = tf.keras.models.load_model(checkpoint_path, compile=False)
    print(f"Loaded model from {checkpoint_path}")

    # Evaluate
    reconstructions = []
    ground_truths = []

    for batch in test_ds:
        # Chỉ truyền ảnh gốc vào model, không cần chỉ số phân loại
        recon = model(batch[0], training=False)  # batch[0] là ảnh đầu vào
        reconstructions.append(recon)
        ground_truths.append(batch[0])  # batch[0] làm ground truth

    reconstructions = tf.concat(reconstructions, axis=0)
    ground_truths = tf.concat(ground_truths, axis=0)

    # Compute average MSE
    mse = tf.reduce_mean(tf.square(reconstructions - ground_truths)).numpy()
    print(f"\nReconstruction MSE on test set: {mse:.6f}\n")

    # Save some reconstructed samples
    plot_reconstruction(ground_truths, reconstructions, save_dir=results['reconstructed_dir'])

    # === Classification via MSE threshold ===
    per_sample_mse = tf.reduce_mean(tf.square(reconstructions - ground_truths), axis=[1, 2, 3]).numpy()
    threshold = eval_cfg['mse_threshold']
    y_pred = (per_sample_mse > threshold).astype(int)

    # Generate labels: test samples are ordered [normal..., abnormal...]
    num_samples = len(per_sample_mse)
    num_per_class = num_samples // 2
    y_true = np.array([0]*num_per_class + [1]*(num_samples - num_per_class))

    # Compute metrics
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Normal", "Abnormal"])

    # Print
    print("=== Classification Results ===")
    print(f"F1-score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    # Save logs
    log_text = (
        f"Reconstruction MSE (avg): {mse:.6f}\n"
        f"MSE Threshold: {threshold}\n"
        f"F1-score: {f1:.4f}\n\n"
        f"Confusion Matrix:\n{cm}\n\n"
        f"Classification Report:\n{report}\n"
    )

    # === ROC Curve ===
    fpr, tpr, _ = roc_curve(y_true, per_sample_mse)  # dùng MSE làm "score"
    auc_score = roc_auc_score(y_true, per_sample_mse)

    # Plot ROC
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    # Save plot
    roc_path = os.path.join(results['evals_dir'], 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved ROC curve to {roc_path}")

    log_text += f"ROC AUC Score: {auc_score:.4f}\n"

    save_log(log_text, log_dir=results['evals_dir'])

if __name__ == "__main__":
    main()
