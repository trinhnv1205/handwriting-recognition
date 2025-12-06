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
import matplotlib
matplotlib.use('qtagg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QTabWidget,
                             QFileDialog, QMessageBox, QFrame, QRadioButton, QButtonGroup,
                             QInputDialog, QComboBox, QScrollArea, QSizePolicy, QGraphicsDropShadowEffect)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QIcon, QCursor
from PyQt6.QtCore import Qt, QPoint, QSize, QPropertyAnimation, QEasingCurve, QTimer

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

    /* Notification */
    QLabel#notification {{
        padding: 10px 15px;
        border-radius: 8px;
        font-weight: bold;
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
        rect = event.rect()
        canvas_painter.drawImage(rect, self.image, rect)

    def get_image(self):
        return self.image

class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.figure.patch.set_facecolor(COLORS["bg"])
        self.ax.set_facecolor(COLORS["bg"])

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def plot_confidence(self, characters, confidences):
        self.ax.clear()

        # Bar chart
        bars = self.ax.bar(characters, confidences, color=COLORS["accent"])

        # Formatting
        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel('Độ tin cậy (%)', color=COLORS["text_secondary"])
        self.ax.tick_params(axis='x', colors=COLORS["text_primary"])
        self.ax.tick_params(axis='y', colors=COLORS["text_secondary"])
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_color(COLORS["border"])
        self.ax.spines['left'].set_color(COLORS["border"])

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.1f}%',
                         ha='center', va='bottom', color=COLORS["text_primary"], fontsize=8)

        self.canvas.draw()

class HandwritingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Studio Nhận Diện Chữ Viết")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(STYLESHEET)

        self.model = None
        self.label_map = None
        self.char_to_index = None
        self.easyocr_reader = None
        self.last_img_cnn = None

        # Notification Timer
        self.notification_timer = QTimer()
        self.notification_timer.setSingleShot(True)
        self.notification_timer.timeout.connect(self.hide_notification)

        self.load_model()
        self.init_ui()

    def show_notification(self, message, type="info"):
        self.lbl_notification.setText(message)
        if type == "success":
            self.lbl_notification.setStyleSheet(f"background-color: {COLORS['success']}; color: white; border-radius: 8px; padding: 10px;")
        elif type == "error":
            self.lbl_notification.setStyleSheet(f"background-color: {COLORS['danger']}; color: white; border-radius: 8px; padding: 10px;")
        elif type == "warning":
            self.lbl_notification.setStyleSheet(f"background-color: {COLORS['warning']}; color: white; border-radius: 8px; padding: 10px;")
        else:
            self.lbl_notification.setStyleSheet(f"background-color: {COLORS['accent']}; color: white; border-radius: 8px; padding: 10px;")

        self.lbl_notification.setVisible(True)
        self.notification_timer.start(3000) # Hide after 3 seconds

    def hide_notification(self):
        self.lbl_notification.setVisible(False)

    def load_model(self):
        try:
            model_path = 'model/handwriting_model.keras'
            label_path = 'model/label_map.json'

            if not os.path.exists(model_path) or not os.path.exists(label_path):
                # We can't use show_notification yet as UI isn't init, so use print or delayed init
                print("Lỗi Hệ Thống: Không tìm thấy file mô hình.")
                return

            self.model = keras.models.load_model(model_path)
            with open(label_path, 'r') as f:
                self.label_map = json.load(f)
            self.label_map = {int(k): v for k, v in self.label_map.items()}
            self.char_to_index = {v: k for k, v in self.label_map.items()}
            print("Hệ thống: Đã tải mô hình.")
        except Exception as e:
            print(f"Lỗi: Không thể tải mô hình: {e}")

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

        # --- Notification Area ---
        self.lbl_notification = QLabel("")
        self.lbl_notification.setObjectName("notification")
        self.lbl_notification.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_notification.setVisible(False)
        main_layout.addWidget(self.lbl_notification)

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
        right_layout.setSpacing(15)

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

        # Chart Widget
        self.chart_widget = MatplotlibWidget()
        self.chart_widget.setFixedHeight(200)
        self.chart_widget.setVisible(False) # Hidden initially
        right_layout.addWidget(self.chart_widget)

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
                self.show_notification("Vui lòng tải ảnh lên trước.", "warning")
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
            except Exception as e:
                self.show_notification(f"Không thể tải ảnh: {e}", "error")

    def save_drawing(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Lưu Ảnh", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
        if file_name:
            if not file_name.endswith(('.png', '.jpg', '.jpeg')):
                file_name += '.png'
            self.canvas.get_image().save(file_name)
            self.show_notification(f"Ảnh đã được lưu tại: {file_name}", "success")

    def segment_characters(self, img_pil):
        # 1. Thresholding
        img_np = np.array(img_pil)
        # Invert if background is white (which is standard for drawing canvas)
        if np.mean(img_np) > 127:
            img_np = 255 - img_np

        # Binary threshold (manual implementation to avoid cv2 dependency if not installed, though numpy is enough)
        # Simple thresholding
        thresh = (img_np > 100).astype(np.uint8) * 255

        # 2. Vertical Projection Profile
        vertical_projection = np.sum(thresh, axis=0)

        # 3. Find gaps
        segments = []
        start = -1
        for i, val in enumerate(vertical_projection):
            if val > 0 and start == -1:
                start = i
            elif val == 0 and start != -1:
                segments.append((start, i))
                start = -1
        if start != -1:
            segments.append((start, len(vertical_projection)))

        # 4. Extract sub-images
        char_images = []
        for (x1, x2) in segments:
            # Add some padding
            x1 = max(0, x1 - 2)
            x2 = min(img_np.shape[1], x2 + 2)

            # Crop
            char_crop = img_np[:, x1:x2]

            # Remove empty rows (top/bottom)
            horiz_proj = np.sum(char_crop, axis=1)
            y_indices = np.where(horiz_proj > 0)[0]
            if len(y_indices) > 0:
                y1, y2 = y_indices[0], y_indices[-1]
                y1 = max(0, y1 - 2)
                y2 = min(img_np.shape[0], y2 + 2)
                char_crop = char_crop[y1:y2, :]

            # Convert back to PIL and preprocess
            char_pil = Image.fromarray(char_crop)
            char_images.append(char_pil)

        return char_images

    def preprocess_char(self, img_pil):
        # Resize to fit in 20x20 box while preserving aspect ratio
        img = img_pil.copy()
        img.thumbnail((20, 20), Image.Resampling.LANCZOS)

        # Create 28x28 black canvas
        new_img = Image.new('L', (28, 28), 0) # 0 is black

        # Paste centered
        offset_x = (28 - img.width) // 2
        offset_y = (28 - img.height) // 2
        new_img.paste(img, (offset_x, offset_y))

        return new_img

    def process_prediction(self, img_pil):
        # Prepare images
        img_tess = img_pil.copy() # Black text on white

        # Get selected engine
        engine = self.combo_engine.currentText()

        result_text = ""
        self.btn_retrain.setVisible(False)
        self.chart_widget.setVisible(False)

        if engine == "Custom AI (CNN)":
            result_text = self.predict_cnn(img_pil)
            self.btn_retrain.setVisible(True)
        elif engine == "Tesseract OCR":
            result_text = self.predict_tesseract(img_tess)
        elif engine == "EasyOCR":
            result_text = self.predict_easyocr(img_tess)

        self.lbl_result.setText(result_text)

    def predict_cnn(self, img_cnn_pil):
        if self.model is None:
            return "Chưa tải mô hình"

        # Segment characters
        char_imgs = self.segment_characters(img_cnn_pil)

        if not char_imgs:
            return "Không tìm thấy ký tự"

        full_text = ""
        confidences = []
        chars = []

        self.last_char_imgs = [] # Store all segments

        for char_img in char_imgs:
            # Preprocess
            processed_img = self.preprocess_char(char_img)
            self.last_char_imgs.append(processed_img)

            img_arr = np.array(processed_img)
            img_arr = img_arr.astype('float32') / 255.0
            img_arr = np.expand_dims(img_arr, axis=-1)
            img_arr = np.expand_dims(img_arr, axis=0)

            prediction = self.model.predict(img_arr, verbose=0)[0]

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

            char_result = self.label_map.get(predicted_class, "?")
            full_text += char_result

            chars.append(char_result)
            confidences.append(confidence * 100)

        # Update Chart
        self.chart_widget.plot_confidence(chars, confidences)
        self.chart_widget.setVisible(True)

        return f"Kết quả: {full_text}"

    def predict_tesseract(self, img_pil):
        tess_config = '--psm 10' # Treat as single character, might need to change for multi
        if self.rb_num.isChecked():
            tess_config += ' -c tessedit_char_whitelist=0123456789'
        elif self.rb_char.isChecked():
            tess_config += ' -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

        # For multi-char, psm 7 or 8 might be better
        tess_config = tess_config.replace('--psm 10', '--psm 7')

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
                return " ".join(results)
            else:
                return "(Không có kết quả)"
        except Exception as e:
            return f"Lỗi: {e}"

    def retrain_model_dialog(self):
        if not hasattr(self, 'last_char_imgs') or not self.last_char_imgs:
            self.show_notification("Không có dữ liệu ảnh gần nhất để dạy.", "warning")
            return

        num_chars = len(self.last_char_imgs)
        text, ok = QInputDialog.getText(self, "Dạy AI", f"Nhập chuỗi ký tự đúng ({num_chars} ký tự):")

        if ok and text:
            correct_text = text.strip()

            if len(correct_text) != num_chars:
                self.show_notification(f"Số lượng ký tự không khớp! Đã nhận diện {num_chars} ảnh, nhưng bạn nhập {len(correct_text)} ký tự.", "error")
                return

            # Validate all characters first
            for char in correct_text:
                if char not in self.char_to_index and char.swapcase() not in self.char_to_index:
                     self.show_notification(f"Ký tự '{char}' chưa được hỗ trợ.", "error")
                     return

            # Save all images
            saved_count = 0
            for i, char in enumerate(correct_text):
                if char not in self.char_to_index:
                    char = char.swapcase()

                label_idx = self.char_to_index[char]
                timestamp = int(time.time()) + i # Add i to ensure unique filenames if fast
                filename = f"user_data/{label_idx}_{timestamp}.png"
                self.last_char_imgs[i].save(filename)
                saved_count += 1

            # Retrain once
            self.perform_retraining()
            self.show_notification(f"AI đã học chuỗi '{correct_text}' thành công!", "success")

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
