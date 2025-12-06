import os
import numpy as np
import tensorflow as keras
from tensorflow.keras import layers, models
import json
import sys

# Add parent directory to path to import train_dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_dataset import load_emnist_data

# Tạo thư mục model nếu chưa tồn tại
if not os.path.exists('model'):
    os.makedirs('model')

# Tải dữ liệu
(images_train, labels_train), (images_test, labels_test) = load_emnist_data()

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
