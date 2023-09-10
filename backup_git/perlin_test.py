#%%
import numpy as np
import matplotlib.pyplot as plt
from noise import snoise2

def generate_perlin_noise(width, height, scale):
    shape = (height, width)
    world = np.zeros(shape)

    for i in range(height):
        for j in range(width):
            x, y = j / scale, i / scale
            world[i][j] = snoise2(x, y, octaves=4, persistence=0.5, lacunarity=2.0, repeatx=1024, repeaty=1024, base=42)

    return world

def apply_perlin_noise_to_image(image, scale, noise_strength):
    noise = generate_perlin_noise(*image.shape, scale)
    noisy_image = image + noise * noise_strength
    return noisy_image

# Load your HAADF-STEM image here. For demonstration purposes, we'll create a random 128x128 numpy array.
haadf_stem_image = np.random.rand(128, 128)

# Set scale and noise_strength to control the appearance of Perlin noise
scale = 100.0
noise_strength = 1000

noisy_image = apply_perlin_noise_to_image(haadf_stem_image, scale, noise_strength)

# Display the original and noisy images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(haadf_stem_image, cmap='gray')
ax1.set_title("Original HAADF-STEM Image")
ax2.imshow(noisy_image, cmap='gray')
ax2.set_title("Noisy Image with Perlin Noise")
plt.show()
# %%
