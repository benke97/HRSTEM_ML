import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import ndimage
import pickle

def align_histograms(histograms):
    ref_hist = histograms[0]
    ref_mode = np.argmax(ref_hist)
    aligned_histograms = [ref_hist]

    for hist in histograms[1:]:
        mode = np.argmax(hist)
        shift = ref_mode - mode
        aligned_histograms.append(ndimage.shift(hist, shift, mode='nearest'))

    return np.array(aligned_histograms)

def average_histograms(images_directory):
    # Read the 32-bit grayscale images from directory
    img_files = [f for f in os.listdir(images_directory) if f.endswith('.tif')]
    histograms = []

    for img_file in img_files:
        img = cv2.imread(os.path.join(images_directory, img_file), cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"Error: Could not read {img_file}")
            continue

        # Subtract the lowest value from the image
        img = img - img.min()

        # Calculate the histogram of the grayscale image
        hist, _ = np.histogram(img.ravel(), bins=256, range=(img.min(), img.max()))
        print(img.min(),img.max())
        histograms.append(hist)

    # Align the histograms
    aligned_histograms = align_histograms(histograms)

    # Sum and average the histograms
    hist_sum = np.sum(aligned_histograms, axis=0)
    avg_hist = hist_sum / len(histograms)

    # Shift the smallest value of the final histogram to zero
    avg_hist = avg_hist - avg_hist.min()

    return avg_hist

def average_histograms_from_pkl(pkl_file_path):
    # Load the list of images from the pickle file
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)

    images = data["images"]
    histograms = []

    for img in images:
        if img is None:
            print("Error: Could not read image")
            continue

        # Subtract the lowest value from the image
        img = img - img.min()

        # Calculate the histogram of the grayscale image
        hist, _ = np.histogram(img.ravel(), bins=256, range=(img.min(), img.max()))
        histograms.append(hist)

    # Align the histograms
    aligned_histograms = align_histograms(histograms)

    # Sum and average the histograms
    hist_sum = np.sum(aligned_histograms, axis=0)
    avg_hist = hist_sum / len(histograms)

    # Shift the smallest value of the final histogram to zero
    avg_hist = avg_hist - avg_hist.min()

    return avg_hist

def plot_histogram(avg_hist):
    plt.figure()
    plt.bar(range(256), avg_hist, color='gray', alpha=0.7)
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.title("Aligned and Averaged Histogram of 32-bit Grayscale Images")
    plt.show()

def plot_histograms(hist1, hist2, color1='gray', color2='blue', label1='Directory', label2='Pickle'):
    plt.figure()
    plt.bar(range(256), hist1, color=color1, alpha=0.7, label=label1)
    plt.bar(range(256), hist2, color=color2, alpha=0.7, label=label2)
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.title("Aligned and Averaged Histogram of 32-bit Grayscale Images")
    plt.legend()
    plt.show()

def plot_histograms_2(hist1, hist2, color1='gray', color2='blue', label1='Directory', label2='Pickle'):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 8))
    
    axes[0].bar(range(256), hist1, color=color1, alpha=0.7)
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"{label1}: Aligned and Averaged Histogram of 32-bit Grayscale Images")
    
    axes[1].bar(range(256), hist2, color=color2, alpha=0.7)
    axes[1].set_xlabel("Intensity")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"{label2}: Aligned and Averaged Histogram of 32-bit Grayscale Images")
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    img_dir = 'data\\experimental_data\\32bit\\'
    #img_dir = 'C:\\Users\\elih\\Documents\\code\\atom_counting'
    avg_histogram = average_histograms(img_dir)
    avg_pkl_histogram = average_histograms_from_pkl('dataset_hist.pkl')
    #print(avg_histogram)
    #plot_histogram(avg_histogram)
    plot_histograms(avg_histogram, avg_pkl_histogram)
    plot_histograms_2(avg_histogram,avg_pkl_histogram)