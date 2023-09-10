import numpy as np
import random
import math
from scipy.ndimage import gaussian_filter
from noise import snoise2
from scipy.spatial import ConvexHull
from skimage.draw import polygon
from skimage import feature
import cv2
from scipy.stats import beta
from skimage.metrics import structural_similarity as ssim
from scipy import spatial
from scipy.ndimage import binary_closing


class Image_Maker:
    def __init__(self):
        self.IMAGE_SIZE = 128
        self.decay_rate = 5
        self.poisson_val = 10
        self.scale = 15
        self.perlin_scale = 10
        self.canvas = np.zeros((self.IMAGE_SIZE,self.IMAGE_SIZE))

    def set_image_size(self,val):
        self.IMAGE_SIZE = val

    def set_scale(self,val):
        self.scale = val

    def set_decay_rate(self,val):
        self.decay_rate = val

    def set_poisson(self,val):
        self.poisson_val = val

    def set_perlin_scale(self,val):
        self.perlin_scale = val

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
        a = random.uniform(-1, 1)
        b = random.uniform(-1, 1)
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

    def generate_distance_map(self, point_set, decay_type='exponential', distance_limit=9):
        points = point_set[['x', 'y']].to_numpy() * self.IMAGE_SIZE
        lattice_constants = point_set[["lattice_constant"]].to_numpy().flatten()
        if decay_type not in ['linear', 'exponential']:
            raise ValueError("Invalid decay type. Expected 'linear' or 'exponential'.")

        distance_map = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE))

        x, y = np.meshgrid(np.arange(self.IMAGE_SIZE), np.arange(self.IMAGE_SIZE))
        x, y = x[:, :, np.newaxis], y[:, :, np.newaxis]
        dist_matrix = np.sqrt((x - points[:, 0])**2 + (y - points[:, 1])**2)

        distance_limits = lattice_constants*random.uniform(50,200)

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

    def generate_perlin_noise_array(self, shape, scale=50.0, octaves=4, persistence=0.2, lacunarity=1.0, contrast=2, power=4):
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

            return noise_map
    
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
        final_mask = np.logical_or(mask_1*0.00001, mask_0)

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
    
    def apply_scan_distortion(self,input_array,distortion_strength):
        """
        Applies smooth (non-discrete) scan distortion to a 2D NumPy array as the scan moves horizontally from the top-left corner.

        Args:
            input_array (np.array): 2D input array of float64
            distortion_strength (float): Strength of the scan distortion (default: 0.05)
            seed (int): Seed for the random number generator (default: None)

        Returns:
            np.array: The input array with scan distortion applied.
        """
        seed=random.randint(1,1337)
        #distortion_strength=random.uniform(0,0.004)
        if seed is not None:
            np.random.seed(seed)

        if len(input_array.shape) != 2:
            raise ValueError("The input array must be a 2D array")

        distorted_array = np.copy(input_array)
        height, width = input_array.shape

        horizontal_distortion = random.choice([True, False])

        if horizontal_distortion:
            for col in range(width):
                # Calculate the distortion amount for the current column
                distortion_amount = np.random.uniform(-distortion_strength * height, distortion_strength * height)

                # Create the original and distorted index arrays for interpolation
                original_indices = np.arange(height)
                distorted_indices = original_indices + distortion_amount

                # Ensure that the distorted_indices are within the valid range
                distorted_indices = np.clip(distorted_indices, 0, height - 1)

                # Apply distortion using linear interpolation
                distorted_array[:, col] = np.interp(original_indices, distorted_indices, input_array[:, col])

        else:
            for row in range(height):
                # Calculate the distortion amount for the current row
                distortion_amount = np.random.uniform(-distortion_strength * width, distortion_strength * width)

                # Create the original and distorted index arrays for interpolation
                original_indices = np.arange(width)
                distorted_indices = original_indices + distortion_amount

                # Ensure that the distorted_indices are within the valid range
                distorted_indices = np.clip(distorted_indices, 0, width - 1)

                # Apply distortion using linear interpolation
                distorted_array[row, :] = np.interp(original_indices, distorted_indices, input_array[row, :])

        return distorted_array
    
    def add_noise(self, image, point_set):
        for lattice_constant in point_set['lattice_constant']:
            if lattice_constant < 0.1 and self.poisson_val < 3:
                self.poisson_val = random.uniform(3,1000)
                break
        hull_mask = cv2.GaussianBlur(cv2.dilate(self.generate_convex_hull_mask(image,point_set).T,np.ones((9, 9), np.uint8)),(9,9),0,0)
        #hull_mask = cv2.dilate(hull_mask,np.ones((5, 5), np.uint8))
        gaussian_noise_image = self.add_gaussian_noise(image, 0, 0.001)
        perlin_noise = gaussian_filter(self.generate_perlin_noise_array(np.shape(image),scale=random.randint(50,300)),0.7)
        speckle = self.generate_perlin_noise_array(np.shape(image),scale=random.randint(1,3))
        small_patches = self.generate_perlin_noise_array(np.shape(image),scale=random.randint(5,20))
        distance_map = self.generate_distance_map(point_set)
        noisy_image = self.apply_poisson_noise(gaussian_noise_image)
        distance_map = self.apply_poisson_noise(perlin_noise*self.perlin_scale*hull_mask+small_patches*hull_mask+distance_map*self.generate_binary_mask(image,0.007),count_scale=50)
        normalized_distance_map = (distance_map - np.min(distance_map)) / np.ptp(distance_map)
        normalized_noisy_image = (noisy_image - np.min(noisy_image)) / np.ptp(noisy_image)
        #print(normalized_distance_map.min(),normalized_distance_map.max())
        if self.poisson_val > 50:
            noisy_image = gaussian_filter(normalized_noisy_image*self.scale+normalized_distance_map+speckle*hull_mask*random.uniform(0,1.5), 0.7)    
        else:
            noisy_image = gaussian_filter(normalized_noisy_image*self.scale+normalized_distance_map+speckle*hull_mask*random.uniform(0,0.01), 0.7)
        distorted_image = self.apply_scan_distortion(noisy_image,random.uniform(0,0.004))

        def skewed_random_float(min_value, max_value, alpha, beta): #beta distribution
            random_float = random.betavariate(alpha, beta)
            return min_value + (max_value - min_value) * random_float
        rand_val = skewed_random_float(0,0.7,1,2)
        if np.random.choice([True,False]):
            distorted_image = np.where((distorted_image < rand_val) & (hull_mask > 0.05), 
                                            np.maximum(hull_mask*np.random.normal(rand_val,0.05, distorted_image.shape),0),
                                            distorted_image)
            distorted_image[normalized_distance_map < 0] = 0 
        noisy_image = self.apply_poisson_noise(distorted_image,count_scale=self.poisson_val)

        #distorted_image[distorted_image<1] = random.uniform(0,1)
        return noisy_image

    def generate_binary_mask(self,input_image,intensity_threshold):
        binary_mask = np.where(input_image > intensity_threshold, 1, 0)
        binary_mask = 1 - binary_mask
        return binary_mask.astype(np.float32)        
    
    def scale_intensities(self, noisy_image):
        # Select a random scale factor between 14000 and 30000
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

    def generate_segmented_image(self, point_set):
        sub_pixel_positions = point_set[['x', 'y']].to_numpy()*self.IMAGE_SIZE
        labels = point_set['label'].to_numpy()
        segmented_image = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE))
        # generate a binary segmented image where each pixel is either 0 or 1 based on the label of the nearest point using KDTree
        # create KDTree once outside the loop
        kdtree = spatial.KDTree(sub_pixel_positions)
        
        for i in range(self.IMAGE_SIZE):
            for j in range(self.IMAGE_SIZE):
                point = np.array([i, j])
                dist, idx = kdtree.query(point)

                if dist > 3 and labels[idx] == 1:
                    segmented_image[i, j] = 0
                else:
                    segmented_image[i, j] = labels[idx]
        segmented_image = segmented_image.T
        #segmented_image = binary_closing(segmented_image, structure=np.ones((3,3))).astype(int)
        return segmented_image

    def generate_image(self,point_set):
        self.set_covariance_matrices(point_set)
        self.set_intensities(point_set)
        self.set_poisson_val(point_set)
        sub_pixel_positions = point_set[['x', 'y']].to_numpy()*self.IMAGE_SIZE
        image = self.generate_bivariate_gaussians(sub_pixel_positions,point_set['Intensity'],covariance_matrices=point_set["Covariance_Matrices"])
        original_image = image.copy()
        noisy_image = self.add_noise(image,point_set)
        print(np.max(noisy_image))
        scaled_image = self.scale_intensities(noisy_image)
        exp_ground_truth = self.generate_exp_gts(sub_pixel_positions)
        gaussian_ground_truth = self.generate_bivariate_gaussians(sub_pixel_positions,point_set['Intensity'],gt=True)  
        segmented_image = self.generate_segmented_image(point_set)
        return noisy_image, gaussian_ground_truth, exp_ground_truth, segmented_image, original_image

    def generate_benchmark_set(self,ideal_image,point_set):
        self.set_poisson(1000)
        noisy_image = self.add_noise_benchmark(ideal_image.copy(),point_set)
        images = [noisy_image]
        


        ssim_list = self.calculate_ssim(images, data_range=1) 
        for ssim in np.arange(0.45,1,0.05)[::-1]:
            #print("ssim",ssim)
            im = self.apply_poisson_noise(noisy_image.copy(),count_scale=self.find_noise_for_ssim(ssim,noisy_image.copy()))
            images.append(im)
        ssim_list = self.calculate_ssim(images, data_range=1)
        # def find_noise_for_ssim(self, target_ssim, ideal_image, point_set, tolerance=0.01, max_iterations=50):
        return images, ssim_list
        
    def find_noise_for_ssim(self, target_ssim, ideal_image, tolerance=0.005, max_iterations=1000):
        low_noise = 1000  # High noise
        high_noise = 0.01  # Low noise
        closest_ssim_diff = float('inf')  # Set to a large value initially
        closest_noise = None

        for iteration in range(max_iterations):
            mid_noise = (low_noise + high_noise) / 2.0
            im = self.apply_poisson_noise(ideal_image.copy(), count_scale=mid_noise)
            ssim_val = self.calculate_ssim([ideal_image,im], data_range=1)[1]

            current_diff = abs(ssim_val - target_ssim)
            
            if current_diff < closest_ssim_diff:
                closest_ssim_diff = current_diff
                closest_noise = mid_noise

            if current_diff < tolerance:
                print("wtf")
                return mid_noise
            elif ssim_val > target_ssim:  # Flip the conditions since higher noise (lower noise value) causes lower SSIM
                low_noise = mid_noise
            else:
                high_noise = mid_noise
            #print(iteration,current_diff,mid_noise)


        print("Reached max iterations without meeting tolerance!", "boi",target_ssim,"bob", closest_ssim_diff)
        return closest_noise

    def generate_unaltered_image(self,point_set):
        self.set_covariance_matrices(point_set)
        self.set_intensities(point_set)
        self.set_poisson_val(point_set)
        sub_pixel_positions = point_set[['x', 'y']].to_numpy()*self.IMAGE_SIZE
        image = self.generate_bivariate_gaussians(sub_pixel_positions,point_set['Intensity'],covariance_matrices=point_set["Covariance_Matrices"])
        return image

    def add_noise_benchmark(self, image, point_set,scan_distort=False):
        for lattice_constant in point_set['lattice_constant']:
            if lattice_constant < 0.1 and self.poisson_val < 3:
                self.poisson_val = random.uniform(3,1000)
                break
        hull_mask = cv2.GaussianBlur(cv2.dilate(self.generate_convex_hull_mask(image,point_set).T,np.ones((9, 9), np.uint8)),(9,9),0,0)
        #hull_mask = cv2.dilate(hull_mask,np.ones((5, 5), np.uint8))
        gaussian_noise_image = self.add_gaussian_noise(image, 0, 0.001)
        perlin_noise = gaussian_filter(self.generate_perlin_noise_array(np.shape(image),scale=random.randint(50,300)),0.7)
        speckle = self.generate_perlin_noise_array(np.shape(image),scale=random.randint(1,3))
        small_patches = self.generate_perlin_noise_array(np.shape(image),scale=random.randint(5,20))
        distance_map = self.generate_distance_map(point_set)
        noisy_image = self.apply_poisson_noise(gaussian_noise_image,count_scale=100000)
        distance_map = self.apply_poisson_noise(perlin_noise*self.perlin_scale*hull_mask+small_patches*hull_mask+distance_map*self.generate_binary_mask(image,0.007),count_scale=50)
        normalized_distance_map = (distance_map - np.min(distance_map)) / np.ptp(distance_map)
        normalized_noisy_image = (noisy_image - np.min(noisy_image)) / np.ptp(noisy_image)
        #print(normalized_distance_map.min(),normalized_distance_map.max())
        if self.poisson_val > 50:
            noisy_image = gaussian_filter(normalized_noisy_image*self.scale+normalized_distance_map+speckle*hull_mask*random.uniform(0,1.5), 0.7)    
        else:
            noisy_image = gaussian_filter(normalized_noisy_image*self.scale+normalized_distance_map+speckle*hull_mask*random.uniform(0,0.01), 0.7)
        distorted_image = noisy_image
        if scan_distort:
            distorted_image = self.apply_scan_distortion(noisy_image,random.uniform(0,0.004))

        def skewed_random_float(min_value, max_value, alpha, beta): #beta distribution
            random_float = random.betavariate(alpha, beta)
            return min_value + (max_value - min_value) * random_float
        rand_val = skewed_random_float(0,0.7,1,2)
        if np.random.choice([True,False]):
            distorted_image = np.where((distorted_image < rand_val) & (hull_mask > 0.05), 
                                            np.maximum(hull_mask*np.random.normal(rand_val,0.05, distorted_image.shape),0),
                                            distorted_image)
            distorted_image[normalized_distance_map < 0] = 0 

        #distorted_image[distorted_image<1] = random.uniform(0,1)
        return distorted_image

    def normalize_image(self,image):
        # Normalize to [0, 1]
        min_val = image.min()
        max_val = image.max()
        return (image - min_val) / (max_val - min_val)

    def calculate_ssim(self, image_list, data_range):
        # Ensure there's at least one image
        if len(image_list) == 0:
            raise ValueError("The image list cannot be empty")

        # Normalize the images
        image_list = [self.normalize_image(img) for img in image_list]

        # Take the first image as the reference
        ref_image = image_list[0]

        # Calculate SSIM for each image in the list
        ssim_values = [ssim(ref_image, img, data_range=data_range, multichannel=False) for img in image_list]

        return ssim_values
    
    def generate_benchmark_images(self,point_set):
        self.set_covariance_matrices(point_set)
        self.set_intensities(point_set)
        self.set_poisson(1000)
        sub_pixel_positions = point_set[['x', 'y']].to_numpy()*self.IMAGE_SIZE
        image = self.generate_bivariate_gaussians(sub_pixel_positions,point_set['Intensity'],covariance_matrices=point_set["Covariance_Matrices"])
        img_distort = self.apply_scan_distortion(image,0.004)
        exp_ground_truth = self.generate_exp_gts(sub_pixel_positions)
        gaussian_ground_truth = self.generate_bivariate_gaussians(sub_pixel_positions,point_set['Intensity'],gt=True)  
        noisy_image = self.add_noise_benchmark(image,point_set)
        distorted_noisy_image = self.apply_scan_distortion(noisy_image,0.004)
        distorted_images = [img_distort,distorted_noisy_image]
        images = [image,noisy_image]
        for noise_val in [1000,500,100,50,10,9,8,7,6,5,4,3,2,1,0.7,0.5]:
            im = self.apply_poisson_noise(noisy_image.copy(),count_scale=noise_val)
            im_d = self.apply_scan_distortion(im,0.004)
            distorted_images.append(im_d)
            images.append(im)
        ssim_list = self.calculate_ssim(images, data_range=1)
        ssim_distorted = self.calculate_ssim(distorted_images,data_range=1)
        return images,distorted_images, gaussian_ground_truth, exp_ground_truth, ssim_list, ssim_distorted