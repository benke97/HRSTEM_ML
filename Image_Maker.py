import numpy as np
import random
import math
from scipy.ndimage import gaussian_filter
from noise import snoise2
from scipy.spatial import ConvexHull
from skimage.draw import polygon
from skimage import feature
import cv2

class Image_Maker:
    def __init__(self):
        self.IMAGE_SIZE = 128
        self.decay_rate = 5
        self.poisson_val = 10
        self.canvas = np.zeros((self.IMAGE_SIZE,self.IMAGE_SIZE))

    def set_image_size(self,val):
        self.IMAGE_SIZE = val
    
    def set_decay_rate(self,val):
        self.decay_rate = val

    def set_poisson(self,val):
        self.poisson_val = val

    def generate_bivariate_gaussians(self, sub_pixel_positions,intensity, covariance_matrices=None,gt=False):
        # Generate meshgrid
        x = np.linspace(0, (self.IMAGE_SIZE - 1), self.IMAGE_SIZE)
        y = np.linspace(0, (self.IMAGE_SIZE - 1), self.IMAGE_SIZE)
        X, Y = np.meshgrid(x, y)

        # Stack meshgrid
        XY = np.stack([X, Y], axis=-1)

        gaussians = []

        for idx, mean in enumerate(sub_pixel_positions):
            mean = np.array(mean)
            
            if covariance_matrices is None:
                # Set default covariance matrix
                covariance_matrix = np.array([[1, 0], [0, 1]])
            else:
                covariance_matrix = covariance_matrices[idx]

            inv_cov = np.linalg.inv(covariance_matrix)
            diff = XY - mean
            exponent = -0.5 * np.einsum("...i,ij,...j", diff, inv_cov, diff)
            if gt:
                gaussian = (1 / (2 * np.pi * np.linalg.det(covariance_matrix))**0.5) * np.exp(exponent)
            else:
                gaussian = (1 / (2 * np.pi * np.linalg.det(covariance_matrix))**0.5) * np.exp(exponent)*intensity[idx]               
            #gaussian /= np.max(gaussian)
            gaussians.append(gaussian)

        # Sum all Gaussians together
        result = np.sum(gaussians, axis=0)

        return result

    def set_covariance_matrices(self,point_set):
        covariance_matrices =  []
        a = random.uniform(-0.7, 0.7)
        b = random.uniform(-0.7, 0.7)
        for index, row in point_set.iterrows():
            lattice_constant = row['lattice_constant']
            if row["lattice_type"] == "hex":
                covariance_matrices.append([[lattice_constant*45, a],[b,lattice_constant*45]])
            elif row["lattice_type"] == "square":
                covariance_matrices.append([[lattice_constant*50, a],[b,lattice_constant*50]])
            elif row["lattice_type"] == "rhombic":
                covariance_matrices.append([[lattice_constant*50, a],[b,lattice_constant*50]])
        point_set['Covariance_Matrices'] = covariance_matrices

    def set_intensities(self,point_set):
        intensities =[]
        decay_k = random.uniform(0,self.decay_rate)
        max_distance_to_interface = point_set['distance_to_interface_center'].max()
        #print(point_set['distance_to_edge'].mean(),point_set['distance_to_edge'].std(),point_set['distance_to_interface_center'].mean(),point_set['distance_to_interface_center'].std())
        for index, row in point_set.iterrows():
            if row['label']:
                def modified_sigmoid(x, decay_k, min_intensity):
                    sigmoid = 1 / (1 + np.exp(decay_k * x))
                    return min_intensity + (1 - min_intensity) * sigmoid

                # Define the decay constant and minimum intensity value
                min_intensity = 0.01

                # Compute the intensity using the modified sigmoid function
                intensity = modified_sigmoid(row['distance_to_interface_center'] / max_distance_to_interface, decay_k, min_intensity)
                #intensity = max(np.exp(-decay_k * (row['distance_to_interface_center'] / max_distance_to_interface))-0.3,0.1)
            else:
                intensity = random.uniform(0.6,0.9)
            intensities.append(intensity)           
        point_set['Intensity'] = intensities

    def set_poisson_val(self,point_set):
        a = [self.poisson_val for _ in range(point_set.shape[0])]
        point_set['Noise_Value'] = a

    def generate_distance_map(self, point_set, decay_type='linear', distance_limit=9):
        points = point_set[['x', 'y']].to_numpy() * self.IMAGE_SIZE
        lattice_constants = point_set[["lattice_constant"]].to_numpy().flatten()
        if decay_type not in ['linear', 'exponential']:
            raise ValueError("Invalid decay type. Expected 'linear' or 'exponential'.")

        distance_map = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE))

        x, y = np.meshgrid(np.arange(self.IMAGE_SIZE), np.arange(self.IMAGE_SIZE))
        x, y = x[:, :, np.newaxis], y[:, :, np.newaxis]
        dist_matrix = np.sqrt((x - points[:, 0])**2 + (y - points[:, 1])**2)

        distance_limits = lattice_constants*100

        if decay_type == 'linear':
            decay_function = np.maximum(distance_limits - dist_matrix, 0) / distance_limits
        else:  # decay_type == 'exponential'
            decay_function = np.exp(-dist_matrix / distance_limits)

        distance_map = np.max(decay_function, axis=2)

        return distance_map

    def apply_poisson_noise(self, array, count_scale=5000):
        scaled_array = array * count_scale
        noisy_array = np.random.poisson(scaled_array)
        noisy_array = noisy_array / count_scale

        return noisy_array

    def add_gaussian_noise(self, image, mean, std):
        noise = np.random.normal(mean, std, image.shape)
        image += noise
        image[image < 0] = 0

        return image

    def add_signal_dependent_gaussian_noise(self, distance_map, noise_scale):
        noise = np.random.normal(0, distance_map * noise_scale)
        noisy_distance_map = distance_map + noise

        return noisy_distance_map
    
    def generate_perlin_noise_array(self, shape, scale=50.0, octaves=4, persistence=0.2, lacunarity=1.0, contrast=1, power=4):
        noise_map = np.zeros(shape, dtype=np.float32)

        for i in range(shape[0]):
            for j in range(shape[1]):
                noise_value = snoise2(
                    i / scale,
                    j / scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=shape[0],
                    repeaty=shape[1],
                    base=42,
                )

                # Normalize the noise value to the range [0, 1]
                noise_value = (noise_value + 1) / 2

                # Apply contrast to the noise value
                noise_value = np.clip((noise_value * contrast) - (contrast - 1) / 2, 0, 1)

                # Apply power transformation to emphasize dark areas
                noise_value = noise_value ** power

                # Apply the noise to the noise_map
                noise_map[i][j] = noise_value

        return gaussian_filter(noise_map,0.7)

    def generate_convex_hull_mask(self, image, point_set):
        # Check if the input image is a valid numpy array and has dtype float64
        if not (isinstance(image, np.ndarray) and image.dtype == np.float64):
            raise ValueError("Input image must be a numpy array with dtype float64")

        # Filter the point_set into two separate arrays based on their labels
        label_1_points = point_set[point_set["label"] == 1][['x', 'y']].values*self.IMAGE_SIZE
        label_0_points = point_set[point_set["label"] == 0][['x', 'y']].values*self.IMAGE_SIZE
        #print(label_0_points,label_1_points)
        # Create a convex hull mask for each filtered array
        mask_1 = self.create_hull_mask(image, label_1_points)
        mask_0 = self.create_hull_mask(image, label_0_points)

        # Combine the two masks by performing a logical OR operation
        final_mask = np.logical_or(mask_1, mask_0)

        return final_mask.astype(np.float64)

    def create_hull_mask(self, image, points):
        # Check if there are enough points to form a convex hull
        if len(points) < 3:
            raise ValueError("Not enough points to form a convex hull")

        # Compute the convex hull
        hull = ConvexHull(points)

        # Extract the vertices of the convex hull
        vertices = points[hull.vertices]
        # Create an empty binary mask with the same shape as the input image
        mask = np.zeros_like(image, dtype=bool)

        # Compute the coordinates of the pixels within the convex hull
        rr, cc = polygon(vertices[:, 0], vertices[:, 1], shape=image.shape)

        # Set the corresponding pixels in the binary mask to True
        mask[rr, cc] = 1

        return mask

    def add_noise(self, image, point_set):
        hull_mask = cv2.GaussianBlur(self.generate_convex_hull_mask(image,point_set).T,(7,7),0,0)
        gaussian_noise_image = self.add_gaussian_noise(image, 0, 0.001)
        perlin_noise = self.generate_perlin_noise_array(np.shape(image))
        distance_map = self.generate_distance_map(point_set)
        noisy_image = self.apply_poisson_noise(gaussian_noise_image)
        distance_map =self.apply_poisson_noise(perlin_noise*5*hull_mask+distance_map*self.generate_binary_mask(image,0.007),count_scale=50)
        normalized_distance_map = (distance_map - np.min(distance_map)) / np.ptp(distance_map)
        normalized_noisy_image = (noisy_image - np.min(noisy_image)) / np.ptp(noisy_image)
        noisy_image = gaussian_filter(normalized_noisy_image*7+normalized_distance_map, 0.7)
        noisy_image = self.apply_poisson_noise(noisy_image,count_scale=self.poisson_val)
        return noisy_image

    def generate_binary_mask(self,input_image,intensity_threshold):
        binary_mask = np.where(input_image > intensity_threshold, 1, 0)
        binary_mask = 1 - binary_mask
        return binary_mask.astype(np.float32)        
    
    def scale_intensities(self, noisy_image):
        # Select a random scale factor between 15000 and 30000
        scale_factor = np.random.randint(14000, 30001)
        #scale_factor = 25000
        normalized_image = self.normalize_image(noisy_image)
        # Scale the intensities of the noisy_image by the scale_factor
        scaled_image = normalized_image * scale_factor

        return scaled_image
    
    def normalize_image(self, image):
        # Find the minimum and maximum pixel values in the image
        min_pixel_value = np.min(image)
        max_pixel_value = np.max(image)

        # Normalize the image to the range [0, 1]
        normalized_image = (image - min_pixel_value) / (max_pixel_value - min_pixel_value)
        return normalized_image

    def generate_exp_gts(self, positions):
        label_image = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE))
        
        for idx, pos in enumerate(positions):
            x, y = pos
            x_int, y_int = int(np.floor(x)), int(np.floor(y))

            # Special cases
            if x_int == 0:
                if y_int == 0:
                    gt_neighborhood = [[x_int, y_int], [x_int + 1, y_int],
                                       [x_int, y_int + 1], [x_int + 1, y_int + 1]]
                elif y_int == 127:
                    gt_neighborhood = [[x_int, y_int - 1], [x_int + 1, y_int - 1],
                                       [x_int, y_int], [x_int + 1, y_int]]
                else:
                    gt_neighborhood = [[x_int, y_int - 1], [x_int + 1, y_int - 1],
                                       [x_int, y_int], [x_int + 1, y_int],
                                       [x_int, y_int + 1], [x_int + 1, y_int + 1]]
            elif x_int == 127:
                if y_int == 0:
                    gt_neighborhood = [[x_int - 1, y_int], [x_int, y_int],
                                       [x_int - 1, y_int + 1], [x_int, y_int + 1]]
                elif y_int == 127:
                    gt_neighborhood = [[x_int - 1, y_int - 1], [x_int, y_int - 1],
                                       [x_int - 1, y_int], [x_int, y_int]]
                else:
                    gt_neighborhood = [[x_int - 1, y_int - 1], [x_int, y_int - 1],
                                       [x_int - 1, y_int], [x_int, y_int],
                                       [x_int - 1, y_int + 1], [x_int, y_int + 1]]
            elif y_int == 0:
                gt_neighborhood = [[x_int - 1, y_int], [x_int, y_int], [x_int + 1, y_int],
                                   [x_int - 1, y_int + 1], [x_int, y_int + 1], [x_int + 1, y_int + 1]]
            elif y_int == 127:
                gt_neighborhood = [[x_int - 1, y_int - 1], [x_int, y_int - 1], [x_int + 1, y_int - 1],
                                   [x_int - 1, y_int], [x_int, y_int], [x_int + 1, y_int]]
            else:
                gt_neighborhood = [[x_int - 1, y_int - 1], [x_int, y_int - 1], [x_int + 1, y_int - 1],
                                   [x_int - 1, y_int], [x_int, y_int], [x_int + 1, y_int],
                                   [x_int - 1, y_int + 1], [x_int, y_int + 1], [x_int + 1, y_int + 1]]

            for idx, pixel in enumerate(gt_neighborhood):
                pixel_center = np.array(pixel) + 0.5
                label_image[pixel[1], pixel[0]] = 1 / (math.pow(math.e, math.dist(pixel_center, pos)))
        return label_image

    def generate_image(self,point_set):
        self.set_covariance_matrices(point_set)
        self.set_intensities(point_set)
        self.set_poisson_val(point_set)
        sub_pixel_positions = point_set[['x', 'y']].to_numpy()*self.IMAGE_SIZE
        image = self.generate_bivariate_gaussians(sub_pixel_positions,point_set['Intensity'],covariance_matrices=point_set["Covariance_Matrices"])
        noisy_image = self.add_noise(image,point_set)
        scaled_image = self.scale_intensities(noisy_image)
        exp_ground_truth = self.generate_exp_gts(sub_pixel_positions)
        gaussian_ground_truth = self.generate_bivariate_gaussians(sub_pixel_positions,point_set['Intensity'],gt=True)  
        return noisy_image, gaussian_ground_truth, exp_ground_truth