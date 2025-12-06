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

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QTabWidget,
                             QFileDialog, QMessageBox, QFrame, QRadioButton, QButtonGroup,
                             QInputDialog, QComboBox, QScrollArea, QSizePolicy, QGraphicsDropShadowEffect)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QIcon, QCursor
from PyQt6.QtCore import Qt, QPoint, QSize, QPropertyAnimation, QEasingCurve

# --- Professional Color Palette (Dracula-inspired but cleaner) ---
COLORS = {
    "bg": "#F4F7FE",          # Light Grey-Blue Background
    "card_bg": "#FFFFFF",     # White Cards
    "text_primary": "#2B3674",# Dark Blue Text
    "text_secondary": "#A3AED0", # Grey Text
    "accent": "#4318FF",      # Bright Blue Accent
    "accent_hover": "#3311CC",
    "success": "#05CD99",     # Green
    "warning": "#FFB547",     # Orange
    "danger": "#EE5D50",      # Red
    "border": "#E0E5F2"       # Light Border
}

STYLESHEET = f"""
    QMainWindow {{
        background-color: {COLORS["bg"]};
    }}
    QWidget {{
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
        color: {COLORS["text_primary"]};
        font-size: 14px;
    }}

    /* Cards */
    QFrame#card {{
        background-color: {COLORS["card_bg"]};
        border-radius: 20px;
        border: 1px solid {COLORS["border"]};
    }}

    /* Buttons */
    QPushButton {{
        background-color: {COLORS["accent"]};
        color: white;
        border-radius: 12px;
        padding: 12px 20px;
        font-weight: 600;
        border: none;
    }}
    QPushButton:hover {{
        background-color: {COLORS["accent_hover"]};
    }}
    QPushButton:pressed {{
        background-color: {COLORS["text_primary"]};
    }}
    QPushButton#secondary {{
        background-color: transparent;
        color: {COLORS["text_primary"]};
        border: 1px solid {COLORS["border"]};
    }}
    QPushButton#secondary:hover {{
        background-color: {COLORS["bg"]};
    }}
    QPushButton#danger {{
        background-color: {COLORS["danger"]};
        color: white;
    }}
    QPushButton#danger:hover {{
        background-color: #D44035;
    }}

    /* Tabs */
    QTabWidget::pane {{
        border: none;
        background: transparent;
    }}
    QTabBar::tab {{
        background: transparent;
        color: {COLORS["text_secondary"]};
        padding: 10px 20px;
        font-weight: 600;
        font-size: 16px;
        border-bottom: 3px solid transparent;
        margin-bottom: 10px;
    }}
    QTabBar::tab:selected {{
        color: {COLORS["accent"]};
        border-bottom: 3px solid {COLORS["accent"]};
    }}
    QTabBar::tab:hover {{
        color: {COLORS["accent"]};
    }}

    /* Inputs */
    QComboBox {{
        background-color: {COLORS["bg"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 10px;
        padding: 8px 12px;
        color: {COLORS["text_primary"]};
    }}
    QComboBox::drop-down {{
        border: none;
    }}

    /* Labels */
    QLabel#header {{
        font-size: 24px;
        font-weight: 700;
        color: {COLORS["text_primary"]};
    }}
    QLabel#subheader {{
        font-size: 14px;
        font-weight: 500;
        color: {COLORS["text_secondary"]};
    }}
    QLabel#result_box {{
        background-color: {COLORS["bg"]};
        border-radius: 15px;
        padding: 20px;
        font-size: 16px;
        color: {COLORS["text_primary"]};
    }}

    /* Canvas */
    QWidget#canvas_wrapper {{
        border: 2px dashed {COLORS["border"]};
        border-radius: 15px;
        background-color: white;
    }}
"""

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
        # Draw the image centered or just top-left? Top-left is fine as we resize image to match widget
        # Actually, we should draw the portion of the image that corresponds to the widget rect
        rect = event.rect()
        canvas_painter.drawImage(rect, self.image, rect)

    def get_image(self):
        return self.image

class HandwritingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Studio Nhận Diện Chữ Viết")
        self.setGeometry(100, 100, 1100, 750)
        self.setStyleSheet(STYLESHEET)

        self.model = None
        self.label_map = None
        self.char_to_index = None
        self.easyocr_reader = None
        self.last_img_cnn = None

        self.load_model()
        self.init_ui()

    def load_model(self):
        try:
            model_path = 'model/handwriting_model.keras'
            label_path = 'model/label_map.json'

            if not os.path.exists(model_path) or not os.path.exists(label_path):
                QMessageBox.critical(self, "Lỗi Hệ Thống", "Không tìm thấy file mô hình. Vui lòng chạy script huấn luyện trước.")
                return

            self.model = keras.models.load_model(model_path)
            with open(label_path, 'r') as f:
                self.label_map = json.load(f)
            self.label_map = {int(k): v for k, v in self.label_map.items()}
            self.char_to_index = {v: k for k, v in self.label_map.items()}
            print("Hệ thống: Đã tải mô hình.")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể tải mô hình: {e}")

    def get_easyocr_reader(self):
        if self.easyocr_reader is None:
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
        return self.easyocr_reader

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        # --- Header ---
        header_layout = QHBoxLayout()
        title_info = QVBoxLayout()
        lbl_title = QLabel("AI Studio Nhận Diện Chữ Viết")
        lbl_title.setObjectName("header")
        lbl_subtitle = QLabel("Hệ thống Nhận diện & Tự học Nâng cao")
        lbl_subtitle.setObjectName("subheader")
        title_info.addWidget(lbl_title)
        title_info.addWidget(lbl_subtitle)
        header_layout.addLayout(title_info)
        header_layout.addStretch()

        # Status Badge (Static for now)
        lbl_status = QLabel("● Hệ thống Sẵn sàng")
        lbl_status.setStyleSheet(f"color: {COLORS['success']}; font-weight: bold; background: {COLORS['bg']}; padding: 8px 15px; border-radius: 15px;")
        header_layout.addWidget(lbl_status)

        main_layout.addLayout(header_layout)

        # --- Main Content Area (2 Columns) ---
        content_layout = QHBoxLayout()

        # Left Column: Workspace (Tabs for Draw/Upload)
        left_card = QFrame()
        left_card.setObjectName("card")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(0, 10, 0, 0)

        self.tabs = QTabWidget()
        self.setup_draw_tab()
        self.setup_upload_tab()
        self.tabs.addTab(self.draw_tab, "Vẽ Tay")
        self.tabs.addTab(self.upload_tab, "Tải Ảnh")

        left_layout.addWidget(self.tabs)
        content_layout.addWidget(left_card, stretch=2)

        # Right Column: Controls & Results
        right_card = QFrame()
        right_card.setObjectName("card")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(25, 25, 25, 25)
        right_layout.setSpacing(20)

        # 1. Configuration Section
        lbl_config = QLabel("Cấu hình")
        lbl_config.setStyleSheet(f"font-weight: bold; color: {COLORS['text_primary']}; font-size: 16px;")
        right_layout.addWidget(lbl_config)

        # Engine Select
        self.combo_engine = QComboBox()
        self.combo_engine.addItems(["Custom AI (CNN)", "Tesseract OCR", "EasyOCR"])
        right_layout.addWidget(QLabel("Bộ Nhận Diện:"))
        right_layout.addWidget(self.combo_engine)

        # Filter Mode
        right_layout.addWidget(QLabel("Chế độ Lọc:"))
        mode_layout = QHBoxLayout()
        self.mode_group = QButtonGroup(self)

        self.rb_all = QRadioButton("Tất cả")
        self.rb_all.setChecked(True)
        self.mode_group.addButton(self.rb_all)
        mode_layout.addWidget(self.rb_all)

        self.rb_num = QRadioButton("Số")
        self.mode_group.addButton(self.rb_num)
        mode_layout.addWidget(self.rb_num)

        self.rb_char = QRadioButton("Chữ")
        self.mode_group.addButton(self.rb_char)
        mode_layout.addWidget(self.rb_char)

        right_layout.addLayout(mode_layout)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"color: {COLORS['border']};")
        right_layout.addWidget(line)

        # 2. Action Section
        lbl_actions = QLabel("Thao tác")
        lbl_actions.setStyleSheet(f"font-weight: bold; color: {COLORS['text_primary']}; font-size: 16px;")
        right_layout.addWidget(lbl_actions)

        btn_predict = QPushButton("Chạy Dự Đoán")
        btn_predict.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_predict.clicked.connect(self.trigger_prediction)
        right_layout.addWidget(btn_predict)

        # 3. Result Section
        right_layout.addStretch()
        lbl_result_title = QLabel("Kết quả Phân tích")
        lbl_result_title.setStyleSheet(f"font-weight: bold; color: {COLORS['text_primary']}; font-size: 16px;")
        right_layout.addWidget(lbl_result_title)

        self.lbl_result = QLabel("Đang chờ đầu vào...")
        self.lbl_result.setObjectName("result_box")
        self.lbl_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_result.setWordWrap(True)
        right_layout.addWidget(self.lbl_result)

        # Retrain Button (Hidden)
        self.btn_retrain = QPushButton("Sửa lỗi & Dạy lại AI")
        self.btn_retrain.setObjectName("danger")
        self.btn_retrain.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_retrain.clicked.connect(self.retrain_model_dialog)
        self.btn_retrain.setVisible(False)
        right_layout.addWidget(self.btn_retrain)

        content_layout.addWidget(right_card, stretch=1)
        main_layout.addLayout(content_layout)

    def setup_draw_tab(self):
        self.draw_tab = QWidget()
        layout = QVBoxLayout(self.draw_tab)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Canvas Wrapper for border
        canvas_wrapper = QWidget()
        canvas_wrapper.setObjectName("canvas_wrapper")
        wrapper_layout = QVBoxLayout(canvas_wrapper)
        wrapper_layout.setContentsMargins(10, 10, 10, 10)

        self.canvas = DrawCanvas()
        wrapper_layout.addWidget(self.canvas)
        layout.addWidget(canvas_wrapper)

        # Canvas Controls
        controls = QHBoxLayout()
        btn_clear = QPushButton("Xóa bảng")
        btn_clear.setObjectName("secondary")
        btn_clear.clicked.connect(self.canvas.clear_image)

        btn_save = QPushButton("Lưu ảnh")
        btn_save.setObjectName("secondary")
        btn_save.clicked.connect(self.save_drawing)

        controls.addStretch()
        controls.addWidget(btn_clear)
        controls.addWidget(btn_save)
        controls.addStretch()

        layout.addLayout(controls)

    def setup_upload_tab(self):
        self.upload_tab = QWidget()
        layout = QVBoxLayout(self.upload_tab)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)

        self.image_display = QLabel("Chưa chọn ảnh")
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display.setFixedSize(400, 400)
        self.image_display.setStyleSheet(f"border: 2px dashed {COLORS['border']}; border-radius: 15px; color: {COLORS['text_secondary']};")

        btn_upload = QPushButton("Chọn Ảnh")
        btn_upload.setObjectName("secondary")
        btn_upload.setFixedWidth(200)
        btn_upload.clicked.connect(self.load_image_file)

        layout.addWidget(self.image_display)
        layout.addWidget(btn_upload)

    def trigger_prediction(self):
        # Determine current tab
        current_idx = self.tabs.currentIndex()
        img = None

        if current_idx == 0: # Draw Tab
            qimage = self.canvas.get_image()
            qimage = qimage.convertToFormat(QImage.Format.Format_RGBA8888)
            width = qimage.width()
            height = qimage.height()
            ptr = qimage.bits()
            ptr.setsize(height * width * 4)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
            img = Image.fromarray(arr).convert('L')
        else: # Upload Tab
            if not hasattr(self, 'uploaded_image') or self.uploaded_image is None:
                QMessageBox.warning(self, "Cảnh báo", "Vui lòng tải ảnh lên trước.")
                return
            img = self.uploaded_image

        self.process_prediction(img)

    def load_image_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn Ảnh", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            try:
                pixmap = QPixmap(file_name)
                scaled_pixmap = pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.image_display.setPixmap(scaled_pixmap)
                self.image_display.setText("")

                self.uploaded_image = Image.open(file_name).convert('L')
                # Auto predict? No, let user click button for consistency in this UI
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể tải ảnh: {e}")

    def save_drawing(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Lưu Ảnh", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
        if file_name:
            if not file_name.endswith(('.png', '.jpg', '.jpeg')):
                file_name += '.png'
            self.canvas.get_image().save(file_name)
            QMessageBox.information(self, "Đã lưu", f"Ảnh đã được lưu tại:\n{file_name}")

    def process_prediction(self, img_pil):
        # Prepare images
        img_tess = img_pil.copy() # Black text on white

        # For CNN: Invert if needed
        np_img = np.array(img_pil)
        if np.mean(np_img) > 127:
            img_cnn = ImageOps.invert(img_pil)
        else:
            img_cnn = img_pil.copy()
        img_cnn = img_cnn.resize((28, 28))

        self.last_img_cnn = img_cnn # Save for retraining

        # Get selected engine
        engine = self.combo_engine.currentText()

        result_text = ""
        self.btn_retrain.setVisible(False)

        if engine == "Custom AI (CNN)":
            result_text = self.predict_cnn(img_cnn)
            self.btn_retrain.setVisible(True)
        elif engine == "Tesseract OCR":
            result_text = self.predict_tesseract(img_tess)
        elif engine == "EasyOCR":
            result_text = self.predict_easyocr(img_tess)

        self.lbl_result.setText(result_text)

    def predict_cnn(self, img_cnn_pil):
        if self.model is None:
            return "Chưa tải mô hình"

        img_arr = np.array(img_cnn_pil)
        img_arr = img_arr.astype('float32') / 255.0
        img_arr = np.expand_dims(img_arr, axis=-1)
        img_arr = np.expand_dims(img_arr, axis=0)

        prediction = self.model.predict(img_arr)[0]

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

        cnn_result = self.label_map.get(predicted_class, "?")
        return f"{cnn_result}\n\nĐộ tin cậy: {confidence*100:.1f}%"

    def predict_tesseract(self, img_pil):
        tess_config = '--psm 10'
        if self.rb_num.isChecked():
            tess_config += ' -c tessedit_char_whitelist=0123456789'
        elif self.rb_char.isChecked():
            tess_config += ' -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

        try:
            tess_result = pytesseract.image_to_string(img_pil, config=tess_config).strip()
            if not tess_result:
                tess_result = "(Không có kết quả)"
            return tess_result
        except Exception:
            return "Lỗi: Không tìm thấy Tesseract"

    def predict_easyocr(self, img_pil):
        try:
            reader = self.get_easyocr_reader()
            allowlist = None
            if self.rb_num.isChecked():
                allowlist = '0123456789'
            elif self.rb_char.isChecked():
                allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

            img_np = np.array(img_pil)
            results = reader.readtext(img_np, detail=0, allowlist=allowlist)

            if results:
                return results[0]
            else:
                return "(Không có kết quả)"
        except Exception as e:
            return f"Lỗi: {e}"

    def retrain_model_dialog(self):
        if self.last_img_cnn is None:
            return

        text, ok = QInputDialog.getText(self, "Dạy AI", "Nhập ký tự đúng (0-9, A-Z, a-z):")
        if ok and text:
            correct_char = text.strip()
            if len(correct_char) != 1:
                QMessageBox.warning(self, "Lỗi", "Vui lòng chỉ nhập đúng 1 ký tự.")
                return

            if correct_char not in self.char_to_index:
                if correct_char.swapcase() in self.char_to_index:
                     correct_char = correct_char.swapcase()
                else:
                    QMessageBox.warning(self, "Lỗi", f"Ký tự '{correct_char}' chưa được hỗ trợ.")
                    return

            label_idx = self.char_to_index[correct_char]

            timestamp = int(time.time())
            filename = f"user_data/{label_idx}_{timestamp}.png"
            self.last_img_cnn.save(filename)

            img_arr = np.array(self.last_img_cnn).astype('float32') / 255.0
            img_arr = np.expand_dims(img_arr, axis=-1)

            self.perform_retraining(target_img_arr=img_arr, target_label=label_idx)

            QMessageBox.information(self, "Thành công", f"AI đã học ký tự '{correct_char}' thành công!")

    def perform_retraining(self, target_img_arr=None, target_label=None):
        images = []
        labels = []
        files = glob.glob("user_data/*.png")

        for f in files:
            try:
                basename = os.path.basename(f)
                label = int(basename.split('_')[0])
                img = Image.open(f).convert('L')
                img = img.resize((28, 28))
                img_arr = np.array(img).astype('float32') / 255.0
                img_arr = np.expand_dims(img_arr, axis=-1)
                images.append(img_arr)
                labels.append(label)
            except Exception:
                continue

        if not images:
            return

        X_train = np.array(images)
        y_train = np.array(labels)

        print(f"Retraining on {len(X_train)} samples...")
        optimizer = keras.optimizers.Adam(learning_rate=0.005)
        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])

        max_attempts = 50
        for i in range(max_attempts):
            self.model.fit(X_train, y_train, epochs=1, verbose=0, batch_size=len(X_train))
            if target_img_arr is not None and target_label is not None:
                pred = self.model.predict(np.expand_dims(target_img_arr, axis=0), verbose=0)
                if np.argmax(pred) == target_label:
                    print(f"Learned in {i+1} epochs.")
                    break

        self.model.save('model/handwriting_model.keras')
        print("Model updated.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandwritingApp()
    window.show()
    sys.exit(app.exec())
