import numpy as np
from PIL import Image

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
