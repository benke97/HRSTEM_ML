import cv2
import matplotlib.pyplot as plt
import numpy as np
def main():
    # Read the 32-bit grayscale image
    
    img_path = 'C:\\Users\\elih\\Documents\\code\\atom_counting\\data\\experimental_data\\32bit\\01.tif'
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print("Error: Could not read the image")
        return
    # Shift the pixel values by the minimum value of the image so that the minimum becomes 0
    img = img - img.min()

    # Define the intensity range
    i_min = 1000
    i_max = 2000

    # Create a binary image where pixels within the intensity range are set to 1 and others are set to 0
    binary_img = np.where((img >= i_min) & (img <= i_max), 1, 0)

    # Plot the binary image
    plt.subplot(1, 2, 1)
    plt.imshow(binary_img*img, cmap='gray')
    plt.title('Binary 32-bit Grayscale Image')

    # Plot the histogram of the grayscale image
    plt.subplot(1, 2, 2)
    plt.hist(img.ravel(), bins=256, range=(img.min(), img.max()), density=False, color='gray', alpha=0.7)
    plt.axvline(i_min, color='r', linestyle='--', label=f'i_min: {i_min}')
    plt.axvline(i_max, color='b', linestyle='--', label=f'i_max: {i_max}')
    plt.legend()
    plt.title('Histogram of 32-bit Grayscale Image')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()