import os
import matplotlib.pyplot as plt
import numpy as np
import yaml

def plot_reconstruction(inputs, outputs, save_dir, prefix='sample', num_samples=5, cmap='gammatone'):
    """
    Vẽ và lưu hình ảnh Groundtruth vs Reconstruction
    """
    os.makedirs(save_dir, exist_ok=True)

    inputs = inputs.numpy() if isinstance(inputs, np.ndarray) == False else inputs
    outputs = outputs.numpy() if isinstance(outputs, np.ndarray) == False else outputs

    for i in range(min(num_samples, inputs.shape[0])):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(inputs[i].squeeze(), cmap=cmap)
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')

        axes[1].imshow(outputs[i].squeeze(), cmap=cmap)
        axes[1].set_title('Reconstruction')
        axes[1].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{prefix}_{i}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")


def model_config(config_path='config/config.yaml'):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def calculate_mse(inputs, outputs):
    """
    Tính Mean Squared Error giữa Groundtruth và Reconstruction
    """
    inputs = inputs.numpy() if isinstance(inputs, np.ndarray) == False else inputs
    outputs = outputs.numpy() if isinstance(outputs, np.ndarray) == False else outputs

    mse = np.mean((inputs - outputs) ** 2)
    return mse

def ensure_dir(dir_path):
    """
    Tạo thư mục nếu chưa tồn tại
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
