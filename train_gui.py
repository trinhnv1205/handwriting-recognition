from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QFrame, QLineEdit, QFileDialog, QMessageBox)
from PyQt6.QtGui import QImage
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
import numpy as np
from PIL import Image
import os
import glob
from tensorflow import keras
import subprocess
import platform
import time
from LetterPreprocessor import preprocess_image, segment_characters
from recognition_gui import DrawCanvas

class RetrainWorker(QThread):
    finished = pyqtSignal(str) # Message
    error = pyqtSignal(str)

    def __init__(self, model, verification_data, user_data_dir):
        super().__init__()
        self.model = model
        self.verification_data = verification_data # List of (img_arr, label)
        self.user_data_dir = user_data_dir

    def run(self):
        try:
            images = []
            labels = []
            files = glob.glob(os.path.join(self.user_data_dir, "*.png"))

            for f in files:
                try:
                    basename = os.path.basename(f)
                    label = int(basename.split('_')[0])
                    img = Image.open(f).convert('L')
                    # Note: Saved images are already preprocessed (28x28)
                    img_arr = np.array(img).astype('float32') / 255.0
                    img_arr = np.expand_dims(img_arr, axis=-1)
                    images.append(img_arr)
                    labels.append(label)
                except Exception:
                    continue

            if not images:
                self.finished.emit("No data to train.")
                return

            X_train = np.array(images)
            y_train = np.array(labels)

            optimizer = keras.optimizers.Adam(learning_rate=0.005)
            self.model.compile(loss='sparse_categorical_crossentropy',
                               optimizer=optimizer,
                               metrics=['accuracy'])

            max_attempts = 50
            learned = False
            for i in range(max_attempts):
                self.model.fit(X_train, y_train, epochs=1, verbose=0, batch_size=len(X_train))

                # Verify
                if self.verification_data:
                    all_correct = True
                    for v_img, v_label in self.verification_data:
                        pred = self.model.predict(np.expand_dims(v_img, axis=0), verbose=0)
                        if np.argmax(pred) != v_label:
                            all_correct = False
                            break

                    if all_correct:
                        self.finished.emit(f"Đã học xong sau {i+1} epochs.")
                        learned = True
                        break

            if not learned:
                self.finished.emit("Đã hoàn tất huấn luyện (Max epochs).")

            self.model.save('model/handwriting_model.keras')

        except Exception as e:
            self.error.emit(str(e))

class TrainingPage(QWidget):
    def __init__(self, model, char_to_index):
        super().__init__()
        self.model = model
        self.char_to_index = char_to_index
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 20, 0, 0)

        # Header
        lbl_title = QLabel("Tạo Dữ Liệu Huấn Luyện")
        lbl_title.setObjectName("header_title")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_title)

        # Body
        body_layout = QHBoxLayout()
        body_layout.setSpacing(30)

        # Left: Canvas
        self.canvas_train = DrawCanvas()
        canvas_container = QFrame()
        canvas_container.setStyleSheet("background-color: white; border-radius: 20px;")
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(10, 10, 10, 10)
        canvas_layout.addWidget(self.canvas_train)

        body_layout.addWidget(canvas_container, stretch=3)

        # Right: Controls
        controls_card = QFrame()
        controls_card.setObjectName("card")
        controls_layout = QVBoxLayout(controls_card)
        controls_layout.setContentsMargins(30, 30, 30, 30)
        controls_layout.setSpacing(20)

        # Label Input
        controls_layout.addWidget(QLabel("Nhãn (ví dụ: 5, A, B)"))
        self.txt_train_label = QLineEdit()
        self.txt_train_label.setPlaceholderText("Nhập nhãn...")
        self.txt_train_label.returnPressed.connect(self.save_and_train)
        controls_layout.addWidget(self.txt_train_label)

        controls_layout.addStretch()

        # Buttons
        btn_save = QPushButton("💾 Lưu ảnh")
        btn_save.setObjectName("btn_green")
        btn_save.clicked.connect(self.save_and_train)
        controls_layout.addWidget(btn_save)

        btn_upload_train = QPushButton("Tải ảnh")
        btn_upload_train.setObjectName("btn_blue")
        btn_upload_train.clicked.connect(self.load_image_train)
        controls_layout.addWidget(btn_upload_train)

        btn_folder = QPushButton("📂 Mở thư mục")
        btn_folder.setObjectName("btn_blue")
        btn_folder.clicked.connect(self.open_user_data_folder)
        controls_layout.addWidget(btn_folder)

        btn_clear = QPushButton("🗑️ Xóa tạm")
        btn_clear.setObjectName("btn_orange")
        btn_clear.clicked.connect(self.canvas_train.clear_image)
        controls_layout.addWidget(btn_clear)

        # Status
        self.lbl_train_status = QLabel("Sẵn sàng vẽ...")
        self.lbl_train_status.setStyleSheet("color: #AAA; font-style: italic;")
        self.lbl_train_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls_layout.addWidget(self.lbl_train_status)

        controls_layout.addStretch()

        body_layout.addWidget(controls_card, stretch=2)
        layout.addLayout(body_layout)

    def save_and_train(self):
        label_text = self.txt_train_label.text().strip()
        print(f"DEBUG: Label text read: '{label_text}'")
        print(f"DEBUG: Label length: {len(label_text)}")
        print(f"DEBUG: Label bytes: {label_text.encode('utf-8')}")

        if not label_text:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập nhãn (ví dụ: A, 5).")
            return

        # Check if characters are valid
        valid_labels = []
        for char in label_text:
            if char not in self.char_to_index:
                if char.swapcase() in self.char_to_index:
                    valid_labels.append(self.char_to_index[char.swapcase()])
                else:
                    QMessageBox.warning(self, "Lỗi", f"Ký tự '{char}' không được hỗ trợ.")
                    return
            else:
                valid_labels.append(self.char_to_index[char])

        # Get image
        qimage = self.canvas_train.get_image()
        qimage = qimage.convertToFormat(QImage.Format.Format_RGBA8888)
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        img_pil = Image.fromarray(arr).convert('L')

        # Segment characters
        if len(label_text) == 1:
            char_imgs = [preprocess_image(img_pil)]
        else:
            char_imgs = segment_characters(img_pil)

            if len(char_imgs) != len(label_text):
                QMessageBox.warning(self, "Lỗi", f"Tìm thấy {len(char_imgs)} ký tự nhưng nhãn có {len(label_text)} ký tự.\nVui lòng viết tách rời nhau.")
                return

        timestamp = int(time.time())
        verification_data = []

        for idx, (char_img, label_idx) in enumerate(zip(char_imgs, valid_labels)):
            filename = f"user_data/{label_idx}_{timestamp}_{idx}.png"
            char_img.save(filename)

            img_arr = np.array(char_img).astype('float32') / 255.0
            img_arr = np.expand_dims(img_arr, axis=-1)
            verification_data.append((img_arr, label_idx))

        self.lbl_train_status.setText("Đang huấn luyện (Vui lòng đợi)...")

        # Start Worker
        self.worker = RetrainWorker(self.model, verification_data, "user_data")
        self.worker.finished.connect(self.on_retrain_finished)
        self.worker.error.connect(self.on_retrain_error)
        self.worker.start()

    def on_retrain_finished(self, message):
        self.lbl_train_status.setText(message)
        self.canvas_train.clear_image()
        self.txt_train_label.clear()
        QTimer.singleShot(3000, lambda: self.lbl_train_status.setText("Sẵn sàng vẽ..."))

    def on_retrain_error(self, error_msg):
        QMessageBox.critical(self, "Lỗi Huấn Luyện", error_msg)
        self.lbl_train_status.setText("Lỗi khi huấn luyện.")

    def open_user_data_folder(self):
        path = os.path.abspath("user_data")
        if not os.path.exists(path):
            os.makedirs(path)

        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def load_image_train(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn Ảnh Huấn Luyện", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            try:
                img = Image.open(file_name).convert('L')
                self.canvas_train.set_image(img)
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể tải ảnh: {e}")
