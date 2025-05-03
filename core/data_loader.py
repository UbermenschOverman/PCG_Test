import tensorflow as tf
import os
import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt

# Hàm load cấu hình từ file YAML
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def preprocess_data(data):
    # Toi cung deo biet tai sao toi lam the nay nua, luc train bi loi nen danh phai xu li the nay: (batch_size, 1387, 17, 3) thanh` (batch_size, 17, 1387, 3)
    data = tf.expand_dims(data, axis=0) 
    data = tf.transpose(data, perm=[0, 2, 1, 3])  # chuyển từ (None, 1387, 17, 3) thành (None, 17, 1387, 3)
    data = tf.squeeze(data, axis=0)

    # Giảm số kênh màu xuống còn 1 (ví dụ: chuyển đổi RGB thành grayscale)
    if data.shape[-1] == 3:
      data = tf.image.rgb_to_grayscale(data)

    return data

# Hàm load và tiền xử lý ảnh, đồng thời sinh nhãn từ tên thư mục
def load_and_preprocess_image_with_label(image_path, img_size):
    img = cv2.imread(image_path.numpy().decode())  # convert Tensor path -> string
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Chuyển đổi img_size thành tuple (width, height) với kiểu int
    width, height = int(img_size[0].numpy()), int(img_size[1].numpy())  # Đảm bảo img_size là int
    img = cv2.resize(img, (width, height))  # Resize ảnh

    img = img / 255.0  # normalize về [0, 1]

    # Sinh label: 0 nếu là ảnh "normal", 1 nếu là "abnormal"
    label = 0 if 'normal' in image_path.numpy().decode() else 1
    return img, label

# Tạo dataset từ thư mục ảnh đã preprocess
def create_dataset(img_size, config_path='config/config.yaml', mode='train'):
    config = load_config(config_path)
    paths = config['data']  
    training_cfg = config['training']
    model_cfg = config['model']

    normal_dir = paths['preprocessed_train_normal_dir']
    abnormal_dir = paths['preprocessed_train_abnormal_dir']
    batch_size = training_cfg['batch_size']
    shuffle = model_cfg['shuffle']

    # Wrapper để dùng với tf.data.map
    def tf_wrapper(image_path):
        img, label = tf.py_function(
            func=load_and_preprocess_image_with_label,
            inp=[image_path, img_size],
            Tout=[tf.float32, tf.int32]
        )
        img.set_shape((img_size[1], img_size[0], 3))  # height, width, channels
        label.set_shape(())
        img = preprocess_data(img)
        return img, label

    if mode == 'train':
        file_dataset = tf.data.Dataset.list_files(str(normal_dir + '/*.png'), shuffle=shuffle)
        dataset = file_dataset.map(tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        val_split = training_cfg['validation_split']
        dataset_size = tf.data.experimental.cardinality(dataset).numpy()
        val_size = int(val_split * dataset_size)

        train_dataset = dataset.skip(val_size)
        val_dataset = dataset.take(val_size)

        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset

    elif mode == 'test':
        normal_dir = paths['preprocessed_test_normal_dir']
        abnormal_dir = paths['preprocessed_test_abnormal_dir']

        normal_files = tf.data.Dataset.list_files(str(normal_dir + '/*.png'), shuffle=shuffle)
        abnormal_files = tf.data.Dataset.list_files(str(abnormal_dir + '/*.png'), shuffle=shuffle)
        all_files = normal_files.concatenate(abnormal_files)

        dataset = all_files.map(tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset

    else:
        raise ValueError("mode phải là 'train' hoặc 'test'")
