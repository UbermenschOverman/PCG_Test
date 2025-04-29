import os
import tensorflow as tf
import yaml

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_label_from_path(file_path):
    # Label: normal -> 0, abnormal -> 1
    parts = tf.strings.split(file_path, os.path.sep)
    class_name = parts[-2]
    return tf.cond(
        tf.equal(class_name, 'normal'),
        lambda: tf.constant(0, dtype=tf.int32),
        lambda: tf.constant(1, dtype=tf.int32)
    )

def load_and_preprocess_image(file_path, img_size=(512, 256)):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, img_size)
    print(f"Image shape after preprocessing: {img.shape}")  
    return img

def create_dataset(img_size, config_path='config/config.yaml', mode='train'):
    config = load_config(config_path)
    paths = config['data']  
    training_cfg = config['training']
    model_config = config['model']

    normal_dir = paths['preprocessed_normal_dir']
    abnormal_dir = paths['preprocessed_abnormal_dir']
    batch_size = training_cfg['batch_size']
    shuffle = model_config['shuffle']

    # Đọc tất cả các file ảnh
    normal_files = tf.data.Dataset.list_files(str(normal_dir + '/*.png'), shuffle=shuffle)
    abnormal_files = tf.data.Dataset.list_files(str(abnormal_dir + '/*.png'), shuffle=shuffle)

    all_files = normal_files.concatenate(abnormal_files)

    # Load và preprocess ảnh và nhãn
    dataset = all_files.map(lambda x: load_and_preprocess_image_with_label(x, img_size=img_size))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    if mode == 'train':
        # Chia dữ liệu thành train và val
        val_split = training_cfg['validation_split']
        dataset_size = tf.data.experimental.cardinality(dataset).numpy()
        val_size = int(val_split * dataset_size)
        
        train_dataset = dataset.skip(val_size)
        val_dataset = dataset.take(val_size)

        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset

    else:
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

def load_and_preprocess_image_with_label(file_path, img_size=(512, 256)):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, img_size)
    label = get_label_from_path(file_path)  # Lấy nhãn từ file_path
    print(f"Image shape after preprocessing: {img.shape}")  
    return img, label

if __name__ == "__main__":
    # Test thử khi chạy trực tiếp
    train_ds, val_ds = create_dataset(mode='train')
    for images, labels in train_ds.take(1):
        print("Images batch shape:", images.shape)
        print("Labels batch shape:", labels.shape)
