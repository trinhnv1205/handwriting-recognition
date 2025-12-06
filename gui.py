import sys
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
import pytesseract
import time
import glob
import easyocr
import subprocess
import platform

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QStackedWidget,
                             QFileDialog, QMessageBox, QFrame, QRadioButton, QButtonGroup,
                             QLineEdit, QProgressBar)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QIcon, QCursor
from PyQt6.QtCore import Qt, QPoint, QSize, QTimer, QThread, pyqtSignal

# --- Design Colors ---
COLORS = {
    "bg_main": "#212121",       # Dark Grey Background
    "bg_card": "#2C2C2C",       # Slightly lighter card bg
    "bg_input": "#383838",      # Input field bg
    "text_main": "#FFFFFF",
    "text_dim": "#AAAAAA",
    "accent_cyan": "#00E5FF",   # Header Text
    "btn_green": "#00B894",     # Save/Recognize
    "btn_green_hover": "#00A383",
    "btn_blue": "#0984E3",      # Upload/Folder
    "btn_blue_hover": "#0870C0",
    "btn_orange": "#E17055",    # Clear
    "btn_orange_hover": "#D35400",
    "toggle_active": "#00E5FF",
    "toggle_inactive": "#555555"
}

STYLESHEET = f"""
    QMainWindow {{
        background-color: {COLORS["bg_main"]};
    }}
    QWidget {{
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        color: {COLORS["text_main"]};
        font-size: 14px;
    }}

    /* Frames & Cards */
    QFrame#card {{
        background-color: {COLORS["bg_card"]};
        border-radius: 15px;
        border: 1px solid #333;
    }}

    /* Headers */
    QLabel#header_title {{
        font-size: 24px;
        font-weight: bold;
        color: {COLORS["accent_cyan"]};
    }}

    /* Buttons */
    QPushButton {{
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        font-size: 14px;
        border: none;
        color: white;
    }}
    QPushButton#btn_green {{
        background-color: {COLORS["btn_green"]};
    }}
    QPushButton#btn_green:hover {{
        background-color: {COLORS["btn_green_hover"]};
    }}

    QPushButton#btn_blue {{
        background-color: {COLORS["btn_blue"]};
    }}
    QPushButton#btn_blue:hover {{
        background-color: {COLORS["btn_blue_hover"]};
    }}

    QPushButton#btn_orange {{
        background-color: {COLORS["btn_orange"]};
    }}
    QPushButton#btn_orange:hover {{
        background-color: {COLORS["btn_orange_hover"]};
    }}

    /* Toggle Buttons */
    QPushButton#toggle_btn {{
        background-color: {COLORS["toggle_inactive"]};
        color: #DDD;
        border-radius: 15px;
        padding: 5px 20px;
        font-size: 12px;
    }}
    QPushButton#toggle_btn:checked {{
        background-color: {COLORS["toggle_active"]};
        color: #000;
        font-weight: bold;
    }}

    /* Inputs */
    QLineEdit {{
        background-color: {COLORS["bg_input"]};
        border: 1px solid #555;
        border-radius: 5px;
        padding: 10px;
        color: white;
        font-size: 16px;
        text-align: center;
    }}

    /* Radio Buttons */
    QRadioButton {{
        font-size: 14px;
    }}
    QRadioButton::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 9px;
        border: 2px solid #777;
    }}
    QRadioButton::indicator:checked {{
        background-color: {COLORS["accent_cyan"]};
        border-color: {COLORS["accent_cyan"]};
    }}

    /* Result Box */
    QLabel#result_box {{
        background-color: {COLORS["bg_input"]};
        border-radius: 10px;
        padding: 15px;
        font-size: 16px;
        font-weight: bold;
        color: {COLORS["accent_cyan"]};
    }}
"""

def preprocess_image(img_pil):
    """
    Preprocess image for EMNIST model:
    1. Convert to grayscale
    2. Invert if white background
    3. Find bounding box of content
    4. Crop and pad to square
    5. Resize to 28x28 with padding (content ~20x20)
    """
    img_pil = img_pil.convert('L')
    img_arr = np.array(img_pil)

    # Invert if background is white (mean > 127)
    if np.mean(img_arr) > 127:
        img_arr = 255 - img_arr

    # Threshold to find content
    coords = np.argwhere(img_arr > 50)

    if coords.size == 0:
        return img_pil.resize((28, 28))

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    cropped = img_arr[y0:y1, x0:x1]

    # Pad to square
    h, w = cropped.shape
    diff = abs(h - w)
    pad_1 = diff // 2
    pad_2 = diff - pad_1

    if h > w:
        pad_width = ((0,0), (pad_1, pad_2))
    else:
        pad_width = ((pad_1, pad_2), (0,0))

    square = np.pad(cropped, pad_width, mode='constant', constant_values=0)

    # Resize to 20x20 and place in center of 28x28
    img_square = Image.fromarray(square)
    img_square = img_square.resize((20, 20), Image.Resampling.LANCZOS)

    final_img = Image.new('L', (28, 28), 0)
    final_img.paste(img_square, (4, 4))

    return final_img

class RetrainWorker(QThread):
    finished = pyqtSignal(str) # Message
    error = pyqtSignal(str)

    def __init__(self, model, target_img_arr, target_label, user_data_dir):
        super().__init__()
        self.model = model
        self.target_img_arr = target_img_arr
        self.target_label = target_label
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
                if self.target_img_arr is not None:
                    pred = self.model.predict(np.expand_dims(self.target_img_arr, axis=0), verbose=0)
                    if np.argmax(pred) == self.target_label:
                        self.finished.emit(f"Đã học xong sau {i+1} epochs.")
                        learned = True
                        break

            if not learned:
                self.finished.emit("Đã hoàn tất huấn luyện (Max epochs).")

            self.model.save('model/handwriting_model.keras')

        except Exception as e:
            self.error.emit(str(e))

class DrawCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StaticContents)
        self.setMinimumSize(400, 400)
        self.image = QImage(self.size(), QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.white)
        self.drawing = False
        self.last_point = QPoint()
        self.pen_width = 28
        self.pen_color = Qt.GlobalColor.black

    def resizeEvent(self, event):
        if self.width() > self.image.width() or self.height() > self.image.height():
            new_width = max(self.width(), self.image.width())
            new_height = max(self.height(), self.image.height())
            new_image = QImage(new_width, new_height, QImage.Format.Format_RGB32)
            new_image.fill(Qt.GlobalColor.white)
            painter = QPainter(new_image)
            painter.drawImage(QPoint(0, 0), self.image)
            self.image = new_image
            super().resizeEvent(event)

    def clear_image(self):
        self.image.fill(Qt.GlobalColor.white)
        self.update()

    def set_image(self, img_pil):
        # Convert PIL to QImage and draw
        img_pil = img_pil.convert("RGB")
        data = img_pil.tobytes("raw", "RGB")
        qim = QImage(data, img_pil.width, img_pil.height, QImage.Format.Format_RGB888)

        # Scale to fit canvas
        scaled_qim = qim.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        self.clear_image()
        painter = QPainter(self.image)
        # Center image
        x = (self.width() - scaled_qim.width()) // 2
        y = (self.height() - scaled_qim.height()) // 2
        painter.drawImage(x, y, scaled_qim)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.pen_color, self.pen_width,
                                Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                                Qt.PenJoinStyle.RoundJoin))
            current_point = event.position().toPoint()
            painter.drawLine(self.last_point, current_point)
            self.last_point = current_point
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        rect = event.rect()
        canvas_painter.drawImage(rect, self.image, rect)

    def get_image(self):
        return self.image

class ConfidenceBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(220)
        self.confidence = 0.0
        self.label = "?"

    def set_data(self, label, confidence):
        self.label = label
        self.confidence = confidence
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Dimensions
        w = self.width()
        h = self.height()
        bar_width = 60
        bar_max_h = 120
        bar_x = (w - bar_width) // 2

        bar_y_start = 40

        # Draw Bar Border (White)
        painter.setPen(QPen(QColor("white"), 3))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(bar_x, bar_y_start, bar_width, bar_max_h)

        # Draw Bar Fill
        if self.confidence > 0:
            fill_h = int(bar_max_h * self.confidence)
            fill_y = bar_y_start + (bar_max_h - fill_h)

            # Color based on confidence
            color = QColor(COLORS["btn_orange"])
            if self.confidence > 0.8:
                color = QColor(COLORS["btn_green"])
            elif self.confidence > 0.5:
                color = QColor(COLORS["btn_blue"])

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            # Adjust rect to fit inside border
            painter.drawRect(bar_x + 2, fill_y, bar_width - 3, fill_h - 2)

        # Draw Percentage Text (Above bar)
        painter.setPen(QColor(COLORS["text_main"]))
        painter.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        painter.drawText(0, 0, w, 35, Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom, f"{self.confidence*100:.1f}%")

        # Draw Label (Below bar)
        painter.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        painter.drawText(0, bar_y_start + bar_max_h + 10, w, 40, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter, self.label)

class HandwritingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Handwriting Recognition Pro v2.0")
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet(STYLESHEET)

        self.model = None
        self.label_map = None
        self.char_to_index = None

        self.load_model()
        self.init_ui()

    def load_model(self):
        try:
            model_path = 'model/handwriting_model.keras'
            label_path = 'model/label_map.json'

            if not os.path.exists(model_path) or not os.path.exists(label_path):
                QMessageBox.critical(self, "Lỗi", "Không tìm thấy model.")
                return

            self.model = keras.models.load_model(model_path)
            with open(label_path, 'r') as f:
                self.label_map = json.load(f)
            self.label_map = {int(k): v for k, v in self.label_map.items()}
            self.char_to_index = {v: k for k, v in self.label_map.items()}
            print("Model loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể tải model: {e}")

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(10)

        # --- Top Navigation (Toggle) ---
        nav_layout = QHBoxLayout()
        nav_layout.addStretch()

        self.btn_mode_recognize = QPushButton("Nhận Diện")
        self.btn_mode_recognize.setObjectName("toggle_btn")
        self.btn_mode_recognize.setCheckable(True)
        self.btn_mode_recognize.setChecked(True)
        self.btn_mode_recognize.clicked.connect(lambda: self.switch_mode(0))

        self.btn_mode_train = QPushButton("Huấn Luyện Mới")
        self.btn_mode_train.setObjectName("toggle_btn")
        self.btn_mode_train.setCheckable(True)
        self.btn_mode_train.clicked.connect(lambda: self.switch_mode(1))

        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.btn_mode_recognize)
        self.mode_group.addButton(self.btn_mode_train)
        self.mode_group.setExclusive(True)

        nav_layout.addWidget(self.btn_mode_recognize)
        nav_layout.addWidget(self.btn_mode_train)
        nav_layout.addStretch()

        main_layout.addLayout(nav_layout)

        # --- Main Content Stack ---
        self.stack = QStackedWidget()

        # Page 1: Recognition
        self.page_recognize = QWidget()
        self.setup_recognition_ui()
        self.stack.addWidget(self.page_recognize)

        # Page 2: Training
        self.page_train = QWidget()
        self.setup_training_ui()
        self.stack.addWidget(self.page_train)

        main_layout.addWidget(self.stack)

    def switch_mode(self, index):
        self.stack.setCurrentIndex(index)

    # ==========================================
    # RECOGNITION UI
    # ==========================================
    def setup_recognition_ui(self):
        layout = QVBoxLayout(self.page_recognize)
        layout.setContentsMargins(0, 20, 0, 0)

        # Header
        lbl_title = QLabel("AI Nhận Diện Chữ Viết Tay")
        lbl_title.setObjectName("header_title")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_title)

        # Body
        body_layout = QHBoxLayout()
        body_layout.setSpacing(30)

        # Left: Canvas
        self.canvas_recog = DrawCanvas()
        # Wrap canvas in a frame to give it rounded corners visual if needed,
        # but DrawCanvas paints white. Let's just add it.
        # To make it look like the screenshot (rounded white box), we can put it in a container.
        canvas_container = QFrame()
        canvas_container.setStyleSheet("background-color: white; border-radius: 20px;")
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(10, 10, 10, 10)
        canvas_layout.addWidget(self.canvas_recog)

        body_layout.addWidget(canvas_container, stretch=3)

        # Right: Controls
        controls_card = QFrame()
        controls_card.setObjectName("card")
        controls_layout = QVBoxLayout(controls_card)
        controls_layout.setContentsMargins(30, 30, 30, 30)
        controls_layout.setSpacing(20)

        # Filter
        filter_layout = QHBoxLayout()
        self.rb_all = QRadioButton("Tất cả")
        self.rb_all.setChecked(True)
        self.rb_num = QRadioButton("Số (0-9)")
        self.rb_char = QRadioButton("Chữ cái (a-z)")

        self.filter_group = QButtonGroup(self)
        self.filter_group.addButton(self.rb_all)
        self.filter_group.addButton(self.rb_num)
        self.filter_group.addButton(self.rb_char)

        filter_layout.addWidget(self.rb_all)
        filter_layout.addWidget(self.rb_num)
        filter_layout.addWidget(self.rb_char)
        controls_layout.addLayout(filter_layout)

        # Result Box
        self.lbl_result = QLabel("KẾT QUẢ NHẬN DIỆN:\n\nCHỮ: ...")
        self.lbl_result.setObjectName("result_box")
        self.lbl_result.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.lbl_result.setFixedHeight(150)
        controls_layout.addWidget(self.lbl_result)

        # Confidence Chart
        self.chart = ConfidenceBar()
        controls_layout.addWidget(self.chart)

        controls_layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()

        btn_upload = QPushButton("Tải ảnh")
        btn_upload.setObjectName("btn_blue")
        btn_upload.clicked.connect(self.load_image_recog)

        btn_save = QPushButton("Lưu ảnh")
        btn_save.setObjectName("btn_blue")
        btn_save.clicked.connect(self.save_drawing_recog)

        btn_predict = QPushButton("Nhận Diện")
        btn_predict.setObjectName("btn_green")
        btn_predict.clicked.connect(self.predict)

        btn_clear = QPushButton("Xóa hết")
        btn_clear.setObjectName("btn_orange")
        btn_clear.clicked.connect(self.clear_recog)

        btn_layout.addWidget(btn_upload)
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_predict)
        btn_layout.addWidget(btn_clear)

        controls_layout.addLayout(btn_layout)

        body_layout.addWidget(controls_card, stretch=2)
        layout.addLayout(body_layout)

    # ==========================================
    # TRAINING UI
    # ==========================================
    def setup_training_ui(self):
        layout = QVBoxLayout(self.page_train)
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

    # ==========================================
    # LOGIC
    # ==========================================
    def clear_recog(self):
        self.canvas_recog.clear_image()
        self.lbl_result.setText("KẾT QUẢ NHẬN DIỆN:\n\nCHỮ: ...")
        self.chart.set_data("?", 0.0)

    def load_image_recog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn Ảnh", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            try:
                img = Image.open(file_name).convert('L')
                self.canvas_recog.set_image(img)
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể tải ảnh: {e}")

    def save_drawing_recog(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Lưu Ảnh", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
        if file_name:
            if not file_name.endswith(('.png', '.jpg', '.jpeg')):
                file_name += '.png'
            self.canvas_recog.get_image().save(file_name)
            QMessageBox.information(self, "Đã lưu", f"Ảnh đã được lưu tại:\n{file_name}")

    def predict(self):
        if self.model is None:
            return

        # Get image from canvas
        qimage = self.canvas_recog.get_image()
        qimage = qimage.convertToFormat(QImage.Format.Format_RGBA8888)
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        img_pil = Image.fromarray(arr).convert('L')

        # Preprocess
        img_cnn = preprocess_image(img_pil)

        img_arr = np.array(img_cnn).astype('float32') / 255.0
        img_arr = np.expand_dims(img_arr, axis=-1)
        img_arr = np.expand_dims(img_arr, axis=0)

        # Predict
        prediction = self.model.predict(img_arr)[0]

        # Filter
        valid_indices = []
        if self.rb_num.isChecked():
            valid_indices = list(range(10))
        elif self.rb_char.isChecked():
            valid_indices = list(range(10, 47))
        else:
            valid_indices = list(range(47))

        masked_prediction = np.copy(prediction)
        mask = np.ones(masked_prediction.shape, dtype=bool)
        mask[valid_indices] = False
        masked_prediction[mask] = -1.0

        predicted_class = np.argmax(masked_prediction)
        confidence = masked_prediction[predicted_class]

        result_char = self.label_map.get(predicted_class, "?")

        # Update UI
        self.lbl_result.setText(f"KẾT QUẢ NHẬN DIỆN:\n\nCHỮ: {result_char}")
        self.chart.set_data(result_char, confidence)

    def save_and_train(self):
        label_text = self.txt_train_label.text().strip()
        print(f"DEBUG: Label text read: '{label_text}'")

        if not label_text:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập nhãn (ví dụ: A, 5).")
            return

        if len(label_text) != 1:
            QMessageBox.warning(self, "Lỗi", "Chỉ nhập đúng 1 ký tự.")
            return

        # Check if label is valid
        if label_text not in self.char_to_index:
            if label_text.swapcase() in self.char_to_index:
                label_text = label_text.swapcase()
            else:
                QMessageBox.warning(self, "Lỗi", f"Ký tự '{label_text}' không được hỗ trợ.")
                return

        label_idx = self.char_to_index[label_text]

        # Get image
        qimage = self.canvas_train.get_image()
        qimage = qimage.convertToFormat(QImage.Format.Format_RGBA8888)
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        img_pil = Image.fromarray(arr).convert('L')

        # Preprocess properly
        img_cnn = preprocess_image(img_pil)

        timestamp = int(time.time())
        filename = f"user_data/{label_idx}_{timestamp}.png"
        img_cnn.save(filename)

        self.lbl_train_status.setText("Đang huấn luyện (Vui lòng đợi)...")

        # Prepare for worker
        img_arr = np.array(img_cnn).astype('float32') / 255.0
        img_arr = np.expand_dims(img_arr, axis=-1)

        # Disable buttons
        self.btn_mode_recognize.setEnabled(False)
        self.btn_mode_train.setEnabled(False)

        # Start Worker
        self.worker = RetrainWorker(self.model, img_arr, label_idx, "user_data")
        self.worker.finished.connect(self.on_retrain_finished)
        self.worker.error.connect(self.on_retrain_error)
        self.worker.start()

    def on_retrain_finished(self, message):
        self.lbl_train_status.setText(message)
        self.canvas_train.clear_image()
        self.txt_train_label.clear()
        self.btn_mode_recognize.setEnabled(True)
        self.btn_mode_train.setEnabled(True)
        QTimer.singleShot(3000, lambda: self.lbl_train_status.setText("Sẵn sàng vẽ..."))

    def on_retrain_error(self, error_msg):
        QMessageBox.critical(self, "Lỗi Huấn Luyện", error_msg)
        self.lbl_train_status.setText("Lỗi khi huấn luyện.")
        self.btn_mode_recognize.setEnabled(True)
        self.btn_mode_train.setEnabled(True)

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandwritingApp()
    window.show()
    sys.exit(app.exec())
