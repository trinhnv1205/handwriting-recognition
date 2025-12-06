import os
import numpy as np
import tensorflow as keras
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
import json

# Tạo thư mục model nếu chưa tồn tại
if not os.path.exists('model'):
    os.makedirs('model')

print("Đang tải dữ liệu EMNIST (balanced) từ tensorflow_datasets...")
# Tải dữ liệu
ds_train, ds_test = tfds.load('emnist/balanced', split=['train', 'test'], shuffle_files=True, as_supervised=True)

# Chuyển đổi sang numpy array để dễ xử lý (hoặc dùng tf.data pipeline)
# Vì dữ liệu không quá lớn, ta có thể load hết vào RAM
def dataset_to_numpy(ds):
    images = []
    labels = []
    for img, label in tfds.as_numpy(ds):
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

print("Converting dataset to numpy...")
images_train, labels_train = dataset_to_numpy(ds_train)
images_test, labels_test = dataset_to_numpy(ds_test)

print(f"Train shape: {images_train.shape}, Labels: {labels_train.shape}")
print(f"Test shape: {images_test.shape}, Labels: {labels_test.shape}")

# EMNIST images in TFDS are also rotated 90 degrees and flipped.
# They come as (28, 28, 1).
# We need to transpose them.
# Transpose (N, 28, 28, 1) -> (N, 28, 28, 1) swapping axis 1 and 2
# Note: images are (N, H, W, C). We want to swap H and W.
images_train = np.transpose(images_train, (0, 2, 1, 3))
images_test = np.transpose(images_test, (0, 2, 1, 3))

# Normalize to 0-1 range
images_train = images_train.astype('float32') / 255.0
images_test = images_test.astype('float32') / 255.0

# Số lượng lớp
num_classes = 47

# Xây dựng mô hình
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Bắt đầu huấn luyện...")
history = model.fit(images_train, labels_train, batch_size=128, epochs=10, validation_split=0.1)

print("Đánh giá mô hình...")
score = model.evaluate(images_test, labels_test, verbose=0)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")

# Lưu mô hình
model_path = 'model/handwriting_model.keras'
model.save(model_path)
print(f"Đã lưu mô hình tại {model_path}")

# Tạo label map
# TFDS EMNIST balanced mapping is consistent with the paper.
label_map = {}
for i in range(47):
    if i < 10:
        label_map[i] = str(i)
    elif i < 36:
        label_map[i] = chr(i - 10 + 65) # A-Z
    else:
        # a, b, d, e, f, g, h, n, q, r, t
        lower_chars = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
        if i - 36 < len(lower_chars):
            label_map[i] = lower_chars[i - 36]
        else:
            label_map[i] = '?'

with open('model/label_map.json', 'w') as f:
    json.dump(label_map, f)
print("Đã lưu label map.")
