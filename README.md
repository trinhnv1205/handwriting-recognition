# Nhận diện chữ viết tay (Handwriting Recognition)

Dự án này sử dụng Python và TensorFlow để nhận diện chữ số và chữ cái viết tay.

## Cài đặt

1.  Đảm bảo bạn đã cài đặt Python 3.
2.  Chạy lệnh sau để cài đặt môi trường:
    ```bash
    # Đã được thực hiện tự động bởi agent
    python3 -m venv venv
    ./venv/bin/pip install -r requirements.txt
    ```

## Sử dụng

1.  **Huấn luyện mô hình** (chỉ cần chạy 1 lần):
    ```bash
    ./venv/bin/python train.py
    ```
    Quá trình này sẽ tải dữ liệu EMNIST và huấn luyện mô hình.

2.  **Chạy ứng dụng**:
    ```bash
    ./run.sh
    ```
    Hoặc:
    ```bash
    ./venv/bin/python gui.py
    ```

## Chạy trên macOS (zsh)

Đây là các bước nhanh để thiết lập và chạy ứng dụng trên macOS (bao gồm Apple Silicon). Bạn có thể sử dụng script helper `run_mac.sh` đã có sẵn trong repository.

1.  Tạo venv, cài dependencies và khởi chạy GUI (copy/paste vào terminal zsh):
    ```bash
    chmod +x run_mac.sh
    ./run_mac.sh
    ```

2.  Nếu bạn muốn thực hiện từng bước thủ công (nếu không dùng script):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    python gui.py
    ```

3.  Apple Silicon (M1/M2) notes:
   - Nếu bạn dùng Apple Silicon và muốn tận dụng Metal GPU, cân nhắc cài:
     ```bash
     pip install tensorflow-macos tensorflow-metal
     ```
   - Repo hiện tại liệt kê `tensorflow` trong `requirements.txt`. Trên macOS/arm64 bạn có thể thay thế hoặc thêm các package trên nếu cần hiệu năng GPU.


## Chức năng
- **Chế độ Vẽ**: Vẽ chữ hoặc số lên bảng trắng và nhấn "Dự đoán".
- **Chế độ Tải ảnh**: Tải ảnh chữ viết tay lên để nhận diện.

## Lưu ý
- Mô hình được huấn luyện trên dữ liệu EMNIST (Balanced), bao gồm chữ số (0-9) và chữ cái (A-Z, a-z).
- Khi vẽ, hãy vẽ nét đậm và rõ ràng để có kết quả tốt nhất.
