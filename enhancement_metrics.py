import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_metrics(original, enhanced):
    mse = np.mean((original - enhanced) ** 2)
    psnr = 10 * np.log10((255 ** 2) / mse) if mse != 0 else float('inf')
    return mse, psnr

def histogram_equalization(image):
    return cv2.equalizeHist(image.astype(np.uint8))

def image_averaging(image, num_images=5):
    averaged = np.zeros_like(image)
    for _ in range(num_images):
        noise = np.random.normal(0, 5, image.shape)
        noisy = image + noise
        averaged += noisy
    averaged = averaged / num_images
    return np.clip(averaged, 0, 255).astype(np.uint8)

def gaussian_smoothing(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def sharpen(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

image = cv2.imread('logo.png' , cv2.IMREAD_GRAYSCALE)
if image is None:
    print("image not found !")
    exit()

image = image.astype(np.float64)

methods = {
    "Histogram Equalization": histogram_equalization(image),
    "Image Averaging": image_averaging(image),
    "Sharpening": sharpen(image),
    "Gaussian smoothing": gaussian_smoothing (image)
}
plt.figure(figsize=(15, 8))
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original")
plt.axis('off')

i = 2
for name, result in methods.items():
    mse, psnr = compute_metrics(image, result)
    print(f"{name} -> MSE: {mse:.2f}, PSNR: {psnr:.2f} dB")

    plt.subplot(2, 3, i)
    plt.imshow(result, cmap='gray')
    plt.title(f"{name}")
    plt.axis('off')
    i += 1

plt.tight_layout()
plt.show()








