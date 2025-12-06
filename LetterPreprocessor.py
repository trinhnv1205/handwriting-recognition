import numpy as np
import cv2
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

def segment_characters(img_pil):
    """
    Segment characters from the image using OpenCV contours.
    Returns a list of preprocessed 28x28 PIL images.
    """
    # Convert to numpy array (grayscale)
    img_pil = img_pil.convert('L')
    img_arr = np.array(img_pil)

    # Invert if background is white (mean > 127)
    if np.mean(img_arr) > 127:
        img_arr = 255 - img_arr

    # Threshold to binary
    _, thresh = cv2.threshold(img_arr, 50, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter small noise
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 5: # Min size
            valid_contours.append((x, y, w, h))

    # Sort contours left to right
    valid_contours.sort(key=lambda c: c[0])

    segmented_images = []
    if not valid_contours:
        # If no contours found, try to use the whole image (fallback)
        segmented_images.append(preprocess_image(img_pil))
    else:
        for (x, y, w, h) in valid_contours:
            # Crop from the original array (which is already inverted if needed)
            # Add some padding around the contour
            pad = 10
            h_img, w_img = img_arr.shape
            y1 = max(0, y - pad)
            y2 = min(h_img, y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(w_img, x + w + pad)

            char_crop = img_arr[y1:y2, x1:x2]

            char_pil = Image.fromarray(char_crop)
            segmented_images.append(preprocess_image(char_pil))

    return segmented_images
