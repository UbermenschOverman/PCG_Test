import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import yaml
import shutil
from data_loader import load_config, load_and_preprocess_image
from model.vqvae import build_vqvae

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)  # Xóa toàn bộ thư mục
    os.makedirs(directory)  # Tạo lại thư mục rỗng

def reconstruct_and_plot(input_image_path, model, save_path=None):
    # Load ảnh và preprocess
    image = load_and_preprocess_image(input_image_path)  # Load ảnh mà không thay đổi data_loader.py
    image = tf.image.resize(image, (128, 128))  # Resize ảnh về kích thước (128, 128)
    image = tf.expand_dims(image, axis=0)  # Thêm batch dimension → (1, 128, 128, 1)
    
    # Dự đoán ảnh tái tạo
    reconstructed = model.predict(image)
    reconstructed = tf.squeeze(reconstructed, axis=0)  # Bỏ batch dimension

    # Hiển thị ảnh gốc và ảnh tái tạo
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(tf.squeeze(image), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(tf.squeeze(reconstructed), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved reconstructed image to: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    # Load config
    config = load_config()
    model_cfg = config['model']
    saved_paths = config['saved_models']
    inference_cfg = config['inference']

    model_path = os.path.join(saved_paths['model_dir'], saved_paths['model_filename'])
    output_dir = inference_cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Build và load model
    model = build_vqvae(
        input_shape=model_cfg['input_shape'],
        embedding_dim=model_cfg['embedding_dim'],
        num_embeddings=model_cfg['num_embeddings'],
        commitment_cost=config['training']['commitment_cost']
    )
    model.load_weights(model_path)
    print(f"Loaded model from {model_path}")

    # Lặp qua 5 ảnh từ tập dữ liệu abnormal để kiểm tra
    input_dir = config['data']['preprocessed_abnormal_dir']
    example_files = [f for f in os.listdir(input_dir) if f.endswith('.png')][:5]

    for file_name in example_files:
        input_path = os.path.join(input_dir, file_name)
        save_path = os.path.join(output_dir, f"reconstructed_{file_name}")
        reconstruct_and_plot(input_path, model, save_path)

if __name__ == "__main__":
    main()
