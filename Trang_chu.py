import sys
import os
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QStackedWidget, QButtonGroup, QMessageBox)
from tensorflow import keras
from recognition_gui import RecognitionPage
from train_gui import TrainingPage

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
        self.page_recognize = RecognitionPage(self.model, self.label_map)
        self.stack.addWidget(self.page_recognize)

        # Page 2: Training
        self.page_train = TrainingPage(self.model, self.char_to_index)
        self.stack.addWidget(self.page_train)

        main_layout.addWidget(self.stack)

    def switch_mode(self, index):
        self.stack.setCurrentIndex(index)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandwritingApp()
    window.show()
    sys.exit(app.exec())
