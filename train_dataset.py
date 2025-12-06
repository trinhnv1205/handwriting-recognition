import tensorflow_datasets as tfds
import numpy as np

def load_emnist_data():
    """
    Load EMNIST balanced dataset and preprocess it.
    Returns: (images_train, labels_train), (images_test, labels_test)
    """
    print("Đang tải dữ liệu EMNIST (balanced) từ tensorflow_datasets...")
    ds_train, ds_test = tfds.load('emnist/balanced', split=['train', 'test'], shuffle_files=True, as_supervised=True)

    print("Converting dataset to numpy...")
    images_train, labels_train = dataset_to_numpy(ds_train)
    images_test, labels_test = dataset_to_numpy(ds_test)

    print(f"Train shape: {images_train.shape}, Labels: {labels_train.shape}")
    print(f"Test shape: {images_test.shape}, Labels: {labels_test.shape}")

    # EMNIST images in TFDS are also rotated 90 degrees and flipped.
    # They come as (28, 28, 1).
    # We need to transpose them.
    # Transpose (N, 28, 28, 1) -> (N, 28, 28, 1) swapping axis 1 and 2
    images_train = np.transpose(images_train, (0, 2, 1, 3))
    images_test = np.transpose(images_test, (0, 2, 1, 3))

    # Normalize to 0-1 range
    images_train = images_train.astype('float32') / 255.0
    images_test = images_test.astype('float32') / 255.0

    return (images_train, labels_train), (images_test, labels_test)

def dataset_to_numpy(ds):
    images = []
    labels = []
    for img, label in tfds.as_numpy(ds):
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)
