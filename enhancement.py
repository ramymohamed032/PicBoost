
import cv2
import numpy as np

def compute_metrics(original, enhanced):
    mse = np.mean((original - enhanced) ** 2)
    psnr = 10 * np.log10((255 ** 2) / mse) if mse != 0 else float('inf')
    return round(mse, 2), round(psnr, 2)

def histogram_equalization(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def image_averaging(image, num_images=5):
    averaged = np.zeros_like(image, dtype=np.float64)
    for _ in range(num_images):
        noise = np.random.normal(0, 5, image.shape)
        noisy = image + noise
        averaged += noisy
    averaged = averaged / num_images
    return np.clip(averaged, 0, 255).astype(np.uint8)

def low_pass_filter(image):
    kernel = np.ones((5, 5), np.float64) / 25
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def gaussian_smoothing(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def remove_salt_and_pepper(image):
     
    if len(image.shape) == 5:
        filtered = cv2.medianBlur(image, 5) 
    else:
        filtered = cv2.medianBlur(image, 5)
    return filtered
