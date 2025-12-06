from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QFrame, QRadioButton, QButtonGroup, QFileDialog, QMessageBox)
from PyQt6.QtGui import QImage, QPainter, QPen, QColor, QFont
from PyQt6.QtCore import Qt, QPoint, QSize
import numpy as np
from PIL import Image
from LetterPreprocessor import segment_characters

COLORS = {
    "bg_main": "#212121",
    "bg_card": "#2C2C2C",
    "bg_input": "#383838",
    "text_main": "#FFFFFF",
    "text_dim": "#AAAAAA",
    "accent_cyan": "#00E5FF",
    "btn_green": "#00B894",
    "btn_green_hover": "#00A383",
    "btn_blue": "#0984E3",
    "btn_blue_hover": "#0870C0",
    "btn_orange": "#E17055",
    "btn_orange_hover": "#D35400",
    "toggle_active": "#00E5FF",
    "toggle_inactive": "#555555"
}

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
        self.data = [] # List of (label, confidence)

    def set_data(self, data):
        """
        data: list of tuples (label, confidence)
        """
        self.data = data
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if not self.data:
            return

        # Dimensions
        w = self.width()
        h = self.height()

        n_bars = len(self.data)
        if n_bars == 0:
            return

        # Calculate bar width and spacing
        # Max bar width 60, min spacing 10
        max_bar_width = 60
        spacing = 20

        total_spacing = (n_bars - 1) * spacing
        total_bar_width = n_bars * max_bar_width

        # If total width exceeds available width, shrink bars or spacing?
        # For now, let's center the group of bars

        group_width = total_bar_width + total_spacing
        start_x = (w - group_width) // 2

        # If start_x is negative (too many bars), we might need to shrink or scroll.
        # For simplicity, assume it fits or just start at 10
        if start_x < 10:
            start_x = 10
            # Recalculate bar width to fit?
            # available_w = w - 20
            # max_bar_width = (available_w - total_spacing) // n_bars

        bar_max_h = 120
        bar_y_start = 40

        for i, (label, confidence) in enumerate(self.data):
            bar_x = start_x + i * (max_bar_width + spacing)

            # Draw Bar Border (White)
            painter.setPen(QPen(QColor("white"), 3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(bar_x, bar_y_start, max_bar_width, bar_max_h)

            # Draw Bar Fill
            if confidence > 0:
                fill_h = int(bar_max_h * confidence)
                fill_y = bar_y_start + (bar_max_h - fill_h)

                # Color based on confidence
                color = QColor(COLORS["btn_orange"])
                if confidence > 0.8:
                    color = QColor(COLORS["btn_green"])
                elif confidence > 0.5:
                    color = QColor(COLORS["btn_blue"])

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                # Adjust rect to fit inside border
                painter.drawRect(bar_x + 2, fill_y, max_bar_width - 3, fill_h - 2)

            # Draw Percentage Text (Above bar)
            painter.setPen(QColor(COLORS["text_main"]))
            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            painter.drawText(bar_x - 10, 0, max_bar_width + 20, 35, Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom, f"{confidence*100:.0f}%")

            # Draw Label (Below bar)
            painter.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
            painter.drawText(bar_x, bar_y_start + bar_max_h + 10, max_bar_width, 40, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter, label)

class RecognitionPage(QWidget):
    def __init__(self, model, label_map):
        super().__init__()
        self.model = model
        self.label_map = label_map
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
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

    def clear_recog(self):
        self.canvas_recog.clear_image()
        self.lbl_result.setText("KẾT QUẢ NHẬN DIỆN:\n\nCHỮ: ...")
        self.chart.set_data([])

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

        char_imgs = segment_characters(img_pil)

        full_result = ""
        confidences = []

        for char_img in char_imgs:
            img_arr = np.array(char_img).astype('float32') / 255.0
            img_arr = np.expand_dims(img_arr, axis=-1)
            img_arr = np.expand_dims(img_arr, axis=0)

            # Predict
            prediction = self.model.predict(img_arr, verbose=0)[0]

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
            full_result += result_char
            confidences.append((result_char, confidence))

        # Update UI
        self.lbl_result.setText(f"KẾT QUẢ NHẬN DIỆN:\n\nCHỮ: {full_result}")

        self.chart.set_data(confidences)
