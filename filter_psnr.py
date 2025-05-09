import cv2
import numpy as np
import matplotlib.pyplot as plt

file_path = 'logo.png'
# تحميل الصورة وتحويلها لرمادي
img = cv2.imread('logo.png', cv2.IMREAD_GRAYSCALE)


# تحويل الصورة لـ float للحسابات
img = img.astype(np.float64)

# إنشاء فلتر منخفض التمرير 5×5
kernel = np.ones((5, 5), np.float64) / 25

# تطبيق الفلتر
filtered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)

# حساب MSE و PSNR
mse = np.mean((img - filtered) ** 2)
psnr = 10 * np.log10((255 ** 2) / mse)

# طباعة النتائج
print(f"MSE: {mse:.2f}")
print(f"PSNR: {psnr:.2f} dB")

# عرض الصور
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Filtered Image (5x5)")
plt.imshow(filtered, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
